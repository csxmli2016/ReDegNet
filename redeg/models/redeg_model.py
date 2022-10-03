import math
import os.path as osp
import torch
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.losses import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.ops import roi_align
from tqdm import tqdm
import cv2
from basicsr.utils import img2tensor
from torchvision.transforms.functional import normalize, rotate
import random
import numpy as np
from basicsr.utils.dist_util import master_only
    

@MODEL_REGISTRY.register()
class ReDegModel(BaseModel):
    """This ReDegModel model for learning real-world degradation from face to natural images"""
    def __init__(self, opt):
        super(ReDegModel, self).__init__(opt)

        # define DegNet network
        self.net_deg = build_network(opt['network_deg'])
        self.net_deg = self.model_to_device(self.net_deg)
        self.print_network(self.net_deg)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_deg', None)
        if load_path is not None:
            self.load_network(self.net_deg, load_path, self.opt['path'].get('strict_load_deg', True))
            print('Congratulations!!!! Successfully load pretrain DegNet models!!!!!!!!!!')

        train_opt = self.opt['train']
        # ----------- define net_syn_d ----------- #
        self.net_synD = build_network(self.opt['network_synD'])
        self.net_synD = self.model_to_device(self.net_synD)
        self.print_network(self.net_synD)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_synD', None)
        if load_path is not None:
            self.load_network(self.net_synD, load_path, self.opt['path'].get('strict_load_synD', True))
        
        # ----------- define net_syn_encoder ----------- #
        self.net_synE = build_network(self.opt['network_synE'])
        self.net_synE = self.model_to_device(self.net_synE)
        self.print_network(self.net_synE)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_synE', None)
        if load_path is not None:
            self.load_network(self.net_synE, load_path, self.opt['path'].get('strict_load_synE', True))
        
        # ----------- define net_syn_generator ----------- #
        self.net_synG = build_network(self.opt['network_synG'])
        self.net_synG = self.model_to_device(self.net_synG)
        self.print_network(self.net_synG)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_synG', None)
        if load_path is not None:
            self.load_network(self.net_synG, load_path, self.opt['path'].get('strict_load_synG', True))

        # ----------- define net_f2n ----------- #
        self.net_f2n = build_network(self.opt['network_f2n'])
        self.net_f2n = self.model_to_device(self.net_f2n)
        self.print_network(self.net_f2n)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_f2n', None)
        if load_path is not None:
            self.load_network(self.net_f2n, load_path, self.opt['path'].get('strict_load_f2n', True))
        
        # ----------- define discrininator of net_f2n ----------- #
        self.net_f2nD = build_network(self.opt['network_f2nD'])
        self.net_f2nD = self.model_to_device(self.net_f2nD)
        self.print_network(self.net_f2nD)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_f2nD', None)
        if load_path is not None:
            self.load_network(self.net_f2nD, load_path, self.opt['path'].get('strict_load_f2nD', True))

        self.net_deg.train()
        self.net_synD.train()
        self.net_synE.train()
        self.net_synG.train()
        self.net_f2n.train()
        self.net_f2nD.train()


        # ----------- define losses for F2N ----------- #
        # pixel loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        # perceptual loss
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        # gan loss
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # ----------- define losses for ReDegNet ----------- #
        if train_opt.get('pixel_syn_opt'):
            self.cri_syn_pix = build_loss(train_opt['pixel_syn_opt']).to(self.device)
        else:
            self.cri_syn_pix = None
        if train_opt.get('perceptual_syn_opt'):
            self.cri_syn_perceptual = build_loss(train_opt['perceptual_syn_opt']).to(self.device)
        else:
            self.cri_syn_perceptual = None

        # gan loss
        self.cri_syn_gan = build_loss(train_opt['gan_syn_opt']).to(self.device)


        # regularization weights
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        self.net_d_reg_every = train_opt['net_d_reg_every']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.prepare_validation()
        self.ShowNum = 4
        self.CheckDeg = None

    def prepare_validation(self):
        # check the test performance during training
        FaceNames = ['000013_face00_scale5.419.png','000013_face02_scale6.106.png']
        WholeNames = ['000013p.png']
        TestPath =  './experiments/validation'
        
        #prepare face pairs for validation
        FixUpScale = 4
        self.TestRealHQ = []
        self.TestRealLQ = []
        for name in FaceNames:
            RealFaceLQTest = cv2.imread(osp.join(TestPath, 'FaceLQ', name), cv2.IMREAD_COLOR)/255.0 #BGR 0~1 
            RealFaceHQTest = cv2.imread(osp.join(TestPath, 'FaceHQ', name), cv2.IMREAD_COLOR)/255.0
            scale = float(name[-9:-4])
            '''
            Note: the scale here referes to the upsample scale factor that the original LQ face resized to HQ size (i.e., 512)
            You can use any robust blind face restoration methods, like GPEN, and GFPGAN.
            '''
            UpSize = int(512 / scale * FixUpScale)# here we consider upsample the lq image to x4
            RealFaceLQTest = cv2.resize(RealFaceLQTest, (UpSize, UpSize))
            RealFaceHQTest = cv2.resize(RealFaceHQTest, (UpSize, UpSize))
            rm_pad = int(UpSize * 0.03)
            RealFaceLQRemovePad = RealFaceLQTest[rm_pad:UpSize-rm_pad, rm_pad:UpSize-rm_pad, :]
            RealFaceHQRemovePad = RealFaceHQTest[rm_pad:UpSize-rm_pad, rm_pad:UpSize-rm_pad, :]
            if min(RealFaceLQRemovePad.shape[:2]) < 256:
                continue
            AfterCropSize, _ = RealFaceHQRemovePad.shape[:2]
            Howmany = math.ceil(AfterCropSize / 256) * 2
            for icrop in range(Howmany):
                w_offset_A = random.randint(0, max(0, AfterCropSize - 256 - 1)) # degradation shift
                h_offset_A = random.randint(0, max(0, AfterCropSize - 256 - 1)) #
                RealFaceLQ = RealFaceLQRemovePad[h_offset_A:h_offset_A+256, w_offset_A:w_offset_A+256,:]
                RealFaceHQ = RealFaceHQRemovePad[h_offset_A:h_offset_A+256, w_offset_A:w_offset_A+256,:]
                RealFaceLQ = img2tensor(RealFaceLQ, bgr2rgb=True, float32=False)
                RealFaceHQ = img2tensor(RealFaceHQ, bgr2rgb=True, float32=False)
                normalize(RealFaceLQ, [0.5,0.5,0.5], [0.5,0.5,0.5], inplace=True) # -1~1 RGB
                normalize(RealFaceHQ, [0.5,0.5,0.5], [0.5,0.5,0.5], inplace=True)
                self.TestRealHQ.append(RealFaceHQ.unsqueeze(0))
                self.TestRealLQ.append(RealFaceLQ.unsqueeze(0))

        self.TestRealHQ = torch.cat(self.TestRealHQ, dim=0).to(self.device)
        self.TestRealLQ = torch.cat(self.TestRealLQ, dim=0).to(self.device)

        #prepare real patches for validation
        self.TestRealPatch = []
        for name in WholeNames:
            RealLQTest = cv2.imread(osp.join(TestPath, 'WholeLQ', name), cv2.IMREAD_COLOR)/255.0 #BGR 0~1 
            h, w = RealLQTest.shape[:2]
            Howmany = math.ceil(min(h,w) / 64) * 2 ** 2
            for icrop in range(Howmany):
                w_offset_A = random.randint(0, max(0, w - 64 - 1)) # test 64->256
                h_offset_A = random.randint(0, max(0, h - 64 - 1)) #
                RealPatch = RealLQTest[h_offset_A:h_offset_A+64, w_offset_A:w_offset_A+64,:]
                RealPatch = img2tensor(RealPatch, bgr2rgb=True, float32=False)
                normalize(RealPatch, [0.5,0.5,0.5], [0.5,0.5,0.5], inplace=True) # -1~1 RGB
                self.TestRealPatch.append(RealPatch.unsqueeze(0))

        self.TestRealPatch = torch.cat(self.TestRealPatch, dim=0).to(self.device)

    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        # ----------- optimizer g ----------- #
        net_g_reg_ratio = 1
        normal_params = []
        for _, param in self.net_f2n.named_parameters():
            normal_params.append(param)
        optim_params_f2n = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_f2n']['lr']
        }]
        optim_type = train_opt['optim_f2n'].pop('type')
        lr = train_opt['optim_f2n']['lr'] * net_g_reg_ratio
        self.optimizer_f2n = self.get_optimizer(optim_type, optim_params_f2n, lr)
        self.optimizers.append(self.optimizer_f2n)

        # ----------- optimizer d ----------- #
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        normal_params = []
        for _, param in self.net_f2nD.named_parameters():
            normal_params.append(param)
        optim_params_f2nD = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_f2nD']['lr']
        }]
        optim_type = train_opt['optim_f2nD'].pop('type')
        lr = train_opt['optim_f2nD']['lr'] * net_d_reg_ratio
        self.optimizer_f2nD = self.get_optimizer(optim_type, optim_params_f2nD, lr)
        self.optimizers.append(self.optimizer_f2nD)

        # ----------- optimizer degnet ----------- #
        #
        normal_params = []
        for _, param in self.net_deg.named_parameters():
            normal_params.append(param)
        optim_params_deg = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_deg']['lr']
        }]
        optim_type = train_opt['optim_deg'].pop('type')
        lr = train_opt['optim_deg']['lr'] * net_g_reg_ratio
        self.optimizer_deg = self.get_optimizer(optim_type, optim_params_deg, lr)
        self.optimizers.append(self.optimizer_deg)

        # ----------- optimizer synencoder ----------- #
        #
        normal_params = []
        for _, param in self.net_synE.named_parameters():
            normal_params.append(param)
        optim_params_synE = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_synE']['lr']
        }]
        optim_type = train_opt['optim_synE'].pop('type')
        lr = train_opt['optim_synE']['lr'] * net_g_reg_ratio
        self.optimizer_synE = self.get_optimizer(optim_type, optim_params_synE, lr)
        self.optimizers.append(self.optimizer_synE)

        # ----------- optimizer syndecoder ----------- #
        # net_g_reg_ratio = 1
        normal_params = []
        for _, param in self.net_synG.named_parameters():
            normal_params.append(param)
        optim_params_synG = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_synG']['lr']
        }]
        optim_type = train_opt['optim_synG'].pop('type')
        lr = train_opt['optim_synG']['lr'] * net_g_reg_ratio
        self.optimizer_synG = self.get_optimizer(optim_type, optim_params_synG, lr)
        self.optimizers.append(self.optimizer_synG)

        # ----------- optimizer syndiscriminator ----------- #
        # net_g_reg_ratio = 1
        normal_params = []
        for _, param in self.net_synD.named_parameters():
            normal_params.append(param)
        optim_params_synD = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_synD']['lr']
        }]
        optim_type = train_opt['optim_synD'].pop('type')
        lr = train_opt['optim_synD']['lr'] * net_g_reg_ratio
        self.optimizer_synD = self.get_optimizer(optim_type, optim_params_synD, lr)
        self.optimizers.append(self.optimizer_synD)


    def feed_data(self, data):
        self.face_real_hq_final = data['face_real_hq_final'].to(self.device)
        self.face_real_lq_final = data['face_real_lq_final'].to(self.device)

        self.face_syn_lq_final = data['face_syn_lq_final'].to(self.device)
        self.natural_syn_lq_final = data['natural_syn_lq_final'].to(self.device)
        self.natural_real_hq_final = data['natural_real_hq_final'].to(self.device)


        self.natural_syn_lq_restore = data['natural_syn_lq_restore'].to(self.device)
        self.natural_real_hq_restore = data['natural_real_hq_restore'].to(self.device)
        self.natural_real_sq_restore = data['natural_real_sq_restore'].to(self.device)

        
    def get_current_visuals(self):
        #f2n
        ShowF2NGT = self.natural_real_hq_restore[:self.ShowNum,:,:,:].detach()
        ShowF2NLQ = F.interpolate(self.natural_syn_lq_restore[:self.ShowNum,:,:,:].detach(), (ShowF2NGT.size(2), ShowF2NGT.size(3)), mode=random.choice(['bilinear', 'bicubic']), align_corners=False)
        ShowF2NResult = self.f2n_output[:self.ShowNum,:,:,:].detach()
        
        #ReDegNet
        ShowFaceRealHQ = self.face_real_hq_final[:self.ShowNum,:,:,:].detach()
        ShowFaceRealLQ = self.face_real_lq_final[:self.ShowNum,:,:,:].detach()
        ShowFaceSynLQ = self.face_syn_lq_final[:self.ShowNum,:,:,:].detach()
        ShowNaturalSynLQ = self.natural_syn_lq_final[:self.ShowNum,:,:,:].detach()
        ShowNaturalRealHQ = self.natural_real_hq_final[:self.ShowNum,:,:,:].detach()

        ShowFaceRealLQFake = self.face_real_lq_fake[:self.ShowNum,:,:,:].detach()
        ShowNaturalSynLQFake = self.natural_syn_lq_fake[:self.ShowNum,:,:,:].detach()
        ShowFaceSynLQFake = self.face_syn_lq_fake[:self.ShowNum,:,:,:].detach()
        
        #validation for face
        index = random.sample(range(self.TestRealLQ.size(0)), self.ShowNum)
        ShowTestFaceLQ = self.TestRealLQ[index,:,:,:].detach()
        ShowTestFaceHQ = self.TestRealHQ[index,:,:,:].detach()
        with torch.no_grad():
            self.net_deg.eval()
            self.net_synE.eval()
            self.net_synG.eval()
            deg = self.net_deg(inputs=torch.cat([ShowTestFaceHQ, ShowTestFaceLQ], 1))
            self.CheckDeg = deg.repeat([3,1]).detach().clone()
            face_content = self.net_synE(inputs=ShowTestFaceHQ)
            syn_face = self.net_synG(styles=[deg[:self.ShowNum,:]], noise=face_content)
            nature_content = self.net_synE(inputs=ShowNaturalRealHQ)
            syn_natural = self.net_synG(styles=[deg[:self.ShowNum,:]], noise=nature_content)
        self.net_deg.train()
        self.net_synE.train()
        self.net_synG.train()

        ShowTestFaceFake = syn_face.detach()
        ShowTestSynNaturalFake = syn_natural.detach()
        
        #validation for specific restoration
        TestNaturalLQ = self.TestRealPatch[random.sample(range(self.TestRealPatch.size(0)), self.ShowNum),:,:,:].detach()
        ShowTestNaturalLQ = F.interpolate(TestNaturalLQ, (ShowF2NGT.size(2), ShowF2NGT.size(3)), mode='bilinear', align_corners=False)
        with torch.no_grad():
            self.net_f2n.eval()
            test_real_natural = self.net_f2n(TestNaturalLQ)
        self.net_f2n.train()
        ShowTestNaturalFake = test_real_natural.detach()
        

        return {'6_F2NGT': ShowF2NGT, '6_F2NLQ': ShowF2NLQ, '6_F2NResult':ShowF2NResult, \
            '2_FaceRealHQ':ShowFaceRealHQ, '2_FaceRealLQ':ShowFaceRealLQ, '2_FaceRealLQFake':ShowFaceRealLQFake, \
            '3_NaturalRealHQ':ShowNaturalRealHQ, '3_NaturalSynLQ':ShowNaturalSynLQ, '3_NaturalSynLQFake':ShowNaturalSynLQFake, \
            '4_1FaceSynLQ':ShowFaceSynLQ, '4_0FaceSynLQFake':ShowFaceSynLQFake, \
            '7_0SynRealNaturalLQ':self.natural_real_syn_fake[:self.ShowNum, :,:,:].detach(), '7_1SynRealNaturalF2N': self.f2n_output2[:self.ShowNum, :,:,:].detach(), \
            '8_1TestFaceLQ':ShowTestFaceLQ, '8_2TestFaceFake':ShowTestFaceFake, '8_0TestFaceHQ':ShowTestFaceHQ, '8_3TestSynNaturalFake':ShowTestSynNaturalFake, \
            '9_TestNaturalLQ':ShowTestNaturalLQ, '9_TestNaturalFake': ShowTestNaturalFake,\
            }


    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        # Training ReDegNet
        for p in self.net_synD.parameters():
            p.requires_grad = False
        self.optimizer_synE.zero_grad()
        self.optimizer_synG.zero_grad()
        self.optimizer_deg.zero_grad()

        self.face_real_hq_content = self.net_synE(inputs=self.face_real_hq_final)
        self.natural_real_hq_content = self.net_synE(inputs=self.natural_real_hq_final)

        self.face_real_deg = self.net_deg(inputs=torch.cat([self.face_real_hq_final, self.face_real_lq_final], 1))
        self.natural_syn_deg = self.net_deg(inputs=torch.cat([self.natural_real_hq_final, self.natural_syn_lq_final], 1))
        self.face_syn_deg = self.net_deg(inputs=torch.cat([self.face_real_hq_final, self.face_syn_lq_final], 1))
        
        self.face_real_lq_fake = self.net_synG(styles=[self.face_real_deg], noise=self.face_real_hq_content)
        if random.random() > 0.2: ##degradation consistent loss with switch operation
            self.natural_syn_lq_fake = self.net_synG(styles=[self.face_syn_deg], noise=self.natural_real_hq_content)
            self.face_syn_lq_fake = self.net_synG(styles=[self.natural_syn_deg], noise=self.face_real_hq_content)
        else:
            self.natural_syn_lq_fake = self.net_synG(styles=[self.natural_syn_deg], noise=self.natural_real_hq_content)
            self.face_syn_lq_fake = self.net_synG(styles=[self.face_syn_deg], noise=self.face_real_hq_content)

        l_redeg_total = 0

        # pixel loss
        l_redeg_pix = self.cri_syn_pix(self.natural_syn_lq_fake, self.natural_syn_lq_final) \
                + self.cri_syn_pix(self.face_syn_lq_fake, self.face_syn_lq_final) \
                + self.cri_syn_pix(self.face_real_lq_fake, self.face_real_lq_final)
        l_redeg_pix = l_redeg_pix * self.opt['train']['syn_pixel_lambda']
        l_redeg_total += l_redeg_pix / 3.0
        loss_dict['l_redeg_pix'] = l_redeg_pix / 3.0

        # perceptual and style loss
        p1, s1 = self.cri_syn_perceptual(self.natural_syn_lq_fake, self.natural_syn_lq_final)
        p2, s2 = self.cri_syn_perceptual(self.face_syn_lq_fake, self.face_syn_lq_final)
        p3, s3 = self.cri_syn_perceptual(self.face_real_lq_fake, self.face_real_lq_final)
        l_redeg_percep = (p1 + p2 + p3)/ 3.0
        l_redeg_style = (s1 + s2 + s3)/ 3.0

        l_redeg_total += l_redeg_percep 
        loss_dict['l_redeg_percep'] = l_redeg_percep

        l_redeg_total += l_redeg_style / 3.0
        loss_dict['l_redeg_style'] = l_redeg_style

        # same degradation contrain
        l_redeg_samedeg = self.cri_syn_pix(self.natural_syn_deg, self.face_syn_deg) * self.opt['train']['syn_samedeg_lambda']
        l_redeg_total += l_redeg_samedeg
        loss_dict['l_redeg_samedeg'] = l_redeg_samedeg

        # contrastive loss:
        l_redeg_contrastive = 1.0 / (self.cri_syn_pix(self.face_real_deg, self.face_syn_deg) + 1e-7) * self.opt['train']['contrastive_lambda']
        
        l_redeg_total += l_redeg_contrastive
        loss_dict['l_redeg_contrastive'] = l_redeg_contrastive

        # gan loss
        fake_synd_pred = self.net_synD(self.face_real_lq_fake, self.face_real_hq_final, self.face_real_deg.detach())
        l_syn_gan = self.cri_syn_gan(fake_synd_pred, True, is_disc=False)
        l_redeg_total += l_syn_gan
        loss_dict['l_redeg_gan'] = l_syn_gan

        l_redeg_total.backward()
        self.optimizer_synE.step() 
        self.optimizer_synG.step()
        self.optimizer_deg.step()

        # ----------- optimize net_synD ----------- #
        for p in self.net_synD.parameters():
            p.requires_grad = True
        self.optimizer_synD.zero_grad()
        
        fake_syn_d_pred = self.net_synD(self.face_real_lq_fake.detach(), self.face_real_hq_final, self.face_real_deg.detach())
        real_syn_d_pred = self.net_synD(self.face_real_lq_final, self.face_real_hq_final, self.face_real_deg.detach())
        l_syn_d = self.cri_syn_gan(real_syn_d_pred, True, is_disc=True) + self.cri_syn_gan(fake_syn_d_pred, False, is_disc=True)
        loss_dict['l_syn_d'] = l_syn_d
        
        loss_dict['real_score_syn'] = real_syn_d_pred.detach().mean()
        loss_dict['fake_score_syn'] = fake_syn_d_pred.detach().mean()
        l_syn_d.backward()
        self.optimizer_synD.step()

        # Training F2N-ESRGAN using pure synthetic degradation 
        if self.opt['train']['train_with_otherDegs']: #(1 is suggested for better generalization ability. You can also set it to 0 for pure synthetic degradation from face pairs, but would have limited generalization ability)
            # ----------- optimize net_f2n ----------- #
            for p in self.net_f2nD.parameters():
                p.requires_grad = False
            self.optimizer_f2n.zero_grad()
            self.f2n_output = self.net_f2n(self.natural_syn_lq_restore)
            l_f2n_total = 0
            l_f2n_pix = self.cri_pix(self.f2n_output, self.natural_real_sq_restore)
            l_f2n_total += l_f2n_pix
            loss_dict['l_f2n_pix'] = l_f2n_pix
            l_f2n_percep, l_f2n_style = self.cri_perceptual(self.f2n_output, self.natural_real_sq_restore)
            if l_f2n_style is None:
                l_f2n_style = 0
            l_f2n_total += l_f2n_percep
            loss_dict['l_f2n_percep'] = l_f2n_percep
            l_f2n_total += l_f2n_style
            loss_dict['l_f2n_style'] = l_f2n_style
            fake_f2n_pred = self.net_f2nD(self.f2n_output)
            l_f2n_gan = self.cri_gan(fake_f2n_pred, True, is_disc=False)
            l_f2n_total += l_f2n_gan
            loss_dict['l_f2n_gan'] = l_f2n_gan
            l_f2n_total.backward()
            self.optimizer_f2n.step()
            # ----------- optimize net_f2nD ----------- #
            for p in self.net_f2nD.parameters():
                p.requires_grad = True
            self.optimizer_f2nD.zero_grad()
            
            fake_d_pred = self.net_f2nD(self.f2n_output.detach())
            real_d_pred = self.net_f2nD(self.natural_real_hq_restore)
            l_f2n_d = self.cri_gan(real_d_pred, True, is_disc=True) + self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_f2n_d'] = l_f2n_d
            #
            loss_dict['real_score_f2n'] = real_d_pred.detach().mean()
            loss_dict['fake_score_f2n'] = fake_d_pred.detach().mean()
            l_f2n_d.backward()
            self.optimizer_f2nD.step()
        else:
            self.f2n_output = self.natural_syn_lq_restore.clone()
        # Training F2N-ESRGAN Again with Synthetic degradation from Face Pairs
        # ----------- optimize net_f2n ----------- #
        for p in self.net_f2nD.parameters():
            p.requires_grad = False
        self.optimizer_f2n.zero_grad()
        
        with torch.no_grad():
            natural_hq_features_again = self.net_synE(inputs=self.natural_real_hq_restore)
            natural_hq_features = []
            for i in natural_hq_features_again:
                natural_hq_features.append(i.detach())
            if self.CheckDeg is not None and random.random() > 0.7:
                deg = self.CheckDeg.detach() # using the degradation from Figure 1
            else:
                deg = self.face_real_deg.detach()

            self.natural_real_syn_fake = self.net_synG(styles=[deg], noise=natural_hq_features)
            # add random noise for avoiding hidden information
            if random.random() > 0.6:
                sigma = random.randint(0,20)/100
                noise = torch.randn(self.natural_real_syn_fake.size()) * sigma
                self.natural_real_syn_fake = torch.clamp(noise.to(self.device) + self.natural_real_syn_fake, -1, 1)

        which_mode = random.random()
        self.f2n_output2 = self.net_f2n(F.interpolate(self.natural_real_syn_fake.detach(), (64,64), mode=random.choice(['bilinear', 'bicubic']), align_corners=False))
        l_f2n_total_2 = 0
        l_f2n_pix_2 = self.cri_pix(self.f2n_output2, self.natural_real_sq_restore)
        l_f2n_total_2 += l_f2n_pix_2
        loss_dict['z_f2n_pix'] = l_f2n_pix_2
        l_f2n_percep_2, l_f2n_style_2 = self.cri_perceptual(self.f2n_output2, self.natural_real_sq_restore)
        if l_f2n_style_2 is None:
            l_f2n_style_2 = 0
        l_f2n_total_2 += l_f2n_percep_2
        loss_dict['z_f2n_percep'] = l_f2n_percep_2
        l_f2n_total_2 += l_f2n_style_2
        loss_dict['z_f2n_style'] = l_f2n_style_2
        fake_f2n_pred = self.net_f2nD(self.f2n_output2)
        l_f2n_gan_2 = self.cri_gan(fake_f2n_pred, True, is_disc=False)
        l_f2n_total_2 += l_f2n_gan_2
        loss_dict['z_f2n_gan'] = l_f2n_gan_2
        l_f2n_total_2.backward()
        self.optimizer_f2n.step()

        # ----------- optimize net_f2nD Again----------- #
        for p in self.net_f2nD.parameters():
            p.requires_grad = True
        self.optimizer_f2nD.zero_grad()
        
        fake_d_pred_2 = self.net_f2nD(self.f2n_output2.detach())
        real_d_pred_2 = self.net_f2nD(self.natural_real_hq_restore)
        l_f2n_d_2 = self.cri_gan(real_d_pred_2, True, is_disc=True) + self.cri_gan(fake_d_pred_2, False, is_disc=True)
        loss_dict['z_f2n_d'] = l_f2n_d_2
        #
        loss_dict['z_real_score_f2n'] = real_d_pred_2.detach().mean()
        loss_dict['z_fake_score_f2n'] = fake_d_pred_2.detach().mean()
        l_f2n_d_2.backward()

        self.optimizer_f2nD.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)


    def save(self, epoch, current_iter):
        # save model
        self.save_network(self.net_deg, 'net_deg', current_iter)
        self.save_network(self.net_synD, 'net_synD', current_iter)
        self.save_network(self.net_synE, 'net_synE', current_iter)
        self.save_network(self.net_synG, 'net_synG', current_iter)
        self.save_network(self.net_f2n, 'net_f2n', current_iter)
        self.save_network(self.net_f2nD, 'net_f2nD', current_iter)
        # save training state
        self.save_training_state(epoch, current_iter)
