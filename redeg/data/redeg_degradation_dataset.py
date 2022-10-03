from cmath import e
from re import T
import cv2
import math
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
import random
from util.real_esrgan_bsrgan_degradation import real_esrgan_degradation, bsrgan_degradation, bsrgan_degradation_plus
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
import time
from PIL import Image, ImageEnhance
import os
import imgaug.augmenters as ia
from util import same_degradation

@DATASET_REGISTRY.register()
class ReDegDegradationDataset(data.Dataset):
    def __init__(self, opt):
        super(ReDegDegradationDataset, self).__init__()
        self.opt = opt
        #
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.dataroot = opt['dataroot']

        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']

        if self.io_backend_opt['type'] == 'disk':
            # dataset 1 df2k
            self.df2kpaths = []
            fp = open(self.opt['df2kpath'], "r")
            lines = fp.read().split("\n")
            lines = [line.strip() for line in lines if len(line)]
            error_flg = False
            for image_file in lines:
                if not osp.exists(image_file):
                    print("%s does not exist!" % (image_file))
                    error_flg = True
                self.df2kpaths.append(image_file)
            if error_flg:
                raise ValueError("Some file paths are corrupted in path {}! Please re-check your file paths!".format(self.opt['df2kpath']))
            fp.close()
            index = np.arange(len(self.df2kpaths))
            np.random.shuffle(index)
            self.df2kpaths = np.array(self.df2kpaths)
            self.df2kpaths = self.df2kpaths[index]

            # dataset 2 FFHQ
            self.ffhqpaths = []
            fp = open(self.opt['ffhqpath'], "r")
            lines = fp.read().split("\n")
            lines = [line.strip() for line in lines if len(line)]
            for image_file in lines:
                if not osp.exists(image_file):
                    print("%s does not exist!" % (image_file))
                    error_flg = True
                self.ffhqpaths.append(image_file)
            if error_flg:
                raise ValueError("Some file paths are corrupted in path {}! Please re-check your file paths!".format(self.opt['ffhqpath']))
            fp.close()
            index = np.arange(len(self.ffhqpaths))
            np.random.shuffle(index)
            self.ffhqpaths = np.array(self.ffhqpaths)
            self.ffhqpaths = self.ffhqpaths[index]

            # dataset 3 real face lq and pairs
            self.reallqpaths = []
            self.realhqpaths = []
            fp = open(self.opt['realpath'], "r") #full path
            lines = fp.read().split("\n")
            lines = [line.strip() for line in lines if len(line)]
            error_flg = False
            for image_file in lines:
                lqpath, hqpath = image_file.split('\t')
                if not osp.exists(lqpath) or not osp.exists(hqpath):
                    print("%s does not exist!" % (lqpath+'_'+hqpath))
                    error_flg = True
                self.reallqpaths.append(lqpath)
                self.realhqpaths.append(hqpath)
            if error_flg:
                raise ValueError("Some file paths are corrupted in path {}! Please re-check your file paths!".format(self.opt['realpath']))
            fp.close()
            
            index = np.arange(len(self.reallqpaths))
            np.random.shuffle(index)
            self.reallqpaths = np.array(self.reallqpaths)
            self.realhqpaths = np.array(self.realhqpaths)
            self.reallqpaths = self.reallqpaths[index]
            self.realhqpaths = self.realhqpaths[index]

            print("[Dataset] Number of Real pairs:", len(self.reallqpaths))
            print("[Dataset] Number of DF2K pairs:", len(self.df2kpaths))
            print("[Dataset] Number of FFHQ pairs:", len(self.ffhqpaths))
            self.paths = self.reallqpaths
        else: 
            raise ValueError('Only support disk for io_backend')

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img


    def AddNoise_Two(self, img, img2=None): # noise
        if img2 is None:
            img2 = img
        if random.random() > 0.9: #
            return img, img2
        sigma = np.random.randint(3, 25)
        img_tensor = torch.from_numpy(np.array(img)).float()
        img2_tensor = torch.from_numpy(np.array(img2)).float()
        noise = torch.randn(img_tensor.size()).mul_(sigma/1.0)
        noiseimg = torch.clamp(noise+img_tensor,0,255)
        noiseimg2 = torch.clamp(noise+img2_tensor,0,255)
        return np.uint8(noiseimg.numpy()), np.uint8(noiseimg2.numpy())

    def AddBlur_Two(self,img, img2=None): # gaussian blur or motion blur
        if img2 is None:
            img2 = img
        if random.random() > 0.8: #
            return img, img2
        blursize = random.randint(3,15) * 2 + 1 ##3,5,7,9,11,13,15
        blursigma = random.randint(3, 13)
        img = cv2.GaussianBlur(img, (blursize,blursize), blursigma/10)
        img2 = cv2.GaussianBlur(img2, (blursize,blursize), blursigma/10)
        return img, img2
    
    def AddDownSample_Two(self,img, img2=None): # downsampling
        if img2 is None:
            img2 = img
        if random.random() > 0.8: #
            return img, img2
        sampler = random.randint(20, 60) * 1.0
        ds_type = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        img = cv2.resize(img, (int(self.out_size/sampler*10.0), int(self.out_size/sampler*10.0)), ds_type)
        img2 = cv2.resize(img2, (int(self.out_size/sampler*10.0), int(self.out_size/sampler*10.0)), ds_type)
        return img, img2
    
    def AddJPEG_Two(self, img, img2=None): # JPEG compression
        if img2 is None:
            img2 = img
        if random.random() > 0.9: #
            return img, img2
        imQ = random.randint(30, 80)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),imQ] # (0,100),higher is better,default is 95
        _, encA = cv2.imencode('.jpg',img,encode_param)
        img = cv2.imdecode(encA,1)

        _, encA2 = cv2.imencode('.jpg',img2,encode_param)
        img2 = cv2.imdecode(encA2,1)
        return img, img2
    
    def AddUpSample_Two(self, img, img2=None):
        if img2 is None:
            img2 = img
        ds_type = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        img = cv2.resize(img, (self.out_size, self.out_size), ds_type)
        img2 = cv2.resize(img2, (self.out_size, self.out_size), ds_type)
        return img, img2

    def usm_sharp(self, img):
        """USM sharpening.
        Input image: I; Blurry image: B.
        1. K = I + weight * (I - B)
        2. Mask = 1 if abs(I - B) > threshold, else: 0
        3. Blur mask:
        4. Out = Mask * K + (1 - Mask) * I
        """
        weight=np.random.randint(3, 7)/10.0#0.5
        radius=np.random.randint(40, 60)#50
        threshold=np.random.randint(7, 15)#10
        if radius % 2 == 0:
            radius += 1
        blur = cv2.GaussianBlur(img, (radius, radius), 0)
        residual = img - blur
        mask = np.abs(residual) * 255 > threshold
        mask = mask.astype('float32')
        soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
        K = img + weight * residual
        K = np.clip(K, 0, 1)
        simg = soft_mask * K + (1 - soft_mask) * img
        return simg

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        #########################################################################################
        ########### Pairs 1: read real face pairs
        #########################################################################################
        reallq_path = self.reallqpaths[index]
        realhq_path = self.realhqpaths[index]
        retry = 3
        while retry > 0:
            try:
                lq_bytes = self.file_client.get(reallq_path)
                hq_bytes = self.file_client.get(realhq_path)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                select_index = random.randint(0, len(self.reallqpaths)-1)
                reallq_path = self.reallqpaths[select_index]
                realhq_path = self.realhqpaths[select_index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        try:
            face_real_lq = imfrombytes(lq_bytes, float32=True)#0~1 BGR
            face_real_hq = imfrombytes(hq_bytes, float32=True)#0~1 BGR
        except:
            print('error in '+reallq_path)
        
        # remove the padding region
        h_s = 60 #height
        w_s = 60 # width
        h_e = 500
        w_e = 450
        face_real_lq_center = face_real_lq[h_s:h_e, w_s:w_e,:]
        face_real_hq_center = face_real_hq[h_s:h_e, w_s:w_e,:]

        # resize 256 ~
        random_scale = random.randint(self.out_size, min(face_real_lq_center.shape[:2])) / min(face_real_lq_center.shape[:2])
        ds_type = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        face_real_lq_center = cv2.resize(face_real_lq_center, (int(random_scale * face_real_lq_center.shape[1]), int(random_scale * face_real_lq_center.shape[0])), ds_type)
        face_real_hq_center = cv2.resize(face_real_hq_center, (int(random_scale * face_real_hq_center.shape[1]), int(random_scale * face_real_hq_center.shape[0])), ds_type)

        h0, w0 = face_real_lq_center.shape[:2]
        ##crop
        h0_offset = random.randint(0, h0 - self.out_size)
        w0_offset = random.randint(0, w0 - self.out_size)
        face_real_lq_center_crop = face_real_lq_center[h0_offset:h0_offset+self.out_size, w0_offset:w0_offset+self.out_size, :]
        face_real_hq_center_crop = face_real_hq_center[h0_offset:h0_offset+self.out_size, w0_offset:w0_offset+self.out_size, :]

        ##flip
        if random.random() > 0.5:
            cv2.flip(face_real_lq_center_crop, 1, face_real_lq_center_crop)
            cv2.flip(face_real_hq_center_crop, 1, face_real_hq_center_crop)
        ##rotate
        if random.random() > 0.65:
            cv2.flip(face_real_lq_center_crop, 0, face_real_lq_center_crop)
            cv2.flip(face_real_hq_center_crop, 0, face_real_hq_center_crop)

        face_real_lq_final = face_real_lq_center_crop.copy()
        face_real_hq_final = face_real_hq_center_crop.copy()

        #degrade the pseudo gt through synthetic degradation
        add_syn_type = random.random()
        if add_syn_type > 0.6:
            ##input should be RGB 0~1 numpy H*W*C
            ##output is RGB 0~1 numpy H*W*C
            gt_tmp = face_real_hq_final[:,:,::-1]#transfer to RGB
            lq_tmp, _ = bsrgan_degradation(gt_tmp, sf=random.choice([3,4,5,6]), lq_patchsize=None)#RGB 0~1 numpy h*w*c
            lq_tmp = cv2.resize(lq_tmp, (self.out_size, self.out_size), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]))
            face_real_lq_final = lq_tmp[:,:,::-1]#transfer to BGR
        elif add_syn_type > 0.35:
            lq_tmp = face_real_lq_final[:,:,::-1]#transfer to RGB
            lq_tmp, _ = bsrgan_degradation(lq_tmp, sf=random.choice([1,2]), lq_patchsize=None)#RGB 0~1 numpy h*w*c
            lq_tmp = cv2.resize(lq_tmp, (self.out_size, self.out_size), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]))
            face_real_lq_final = lq_tmp[:,:,::-1]#transfer to BGR

        if random.random() > 0.95: #gray
            aug_gray = ia.Sometimes(1, ia.Grayscale(alpha=random.choice([0.8, 0.9, 1.0])))
            face_real_hq_final = aug_gray(image=np.uint8(face_real_hq_final*255))/255.0
            face_real_lq_final = aug_gray(image=np.uint8(face_real_lq_final*255))/255.0

        #########################################################################################
        ########### Pairs 2: same degradation on the lq face from pseudo hq face and natural images
        #########################################################################################
        natural_path = self.df2kpaths[random.randint(0, len(self.df2kpaths)-1)]
        retry = 3
        while retry > 0:
            try:
                hq_bytes = self.file_client.get(natural_path)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                #
                natural_path = self.df2kpaths[random.randint(0, len(self.df2kpaths)-1)]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        try:
            natural_real_hq = imfrombytes(hq_bytes, float32=True)#0~1 BGR
        except:
            print('error in '+natural_path)

        
        # resize 256 ~ and crop
        max_size = 360
        random_scale = random.randint(self.out_size, min(list(natural_real_hq.shape[:2])+[max_size])) / min(list(natural_real_hq.shape[:2])+[max_size])
        natural_real_hq = cv2.resize(natural_real_hq, (int(random_scale * natural_real_hq.shape[1]), int(random_scale * natural_real_hq.shape[0])), random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
        h1, w1 = natural_real_hq.shape[:2]
        h1_offset = random.randint(0, h1 - self.out_size)
        w1_offset = random.randint(0, w1 - self.out_size)
        natural_real_hq_final = natural_real_hq[h1_offset:h1_offset+self.out_size, w1_offset:w1_offset+self.out_size, :]

        #color jitter
        if random.random() > 0.6:
            brightness = self.opt.get('brightness', (0.9, 1.1))
            contrast = self.opt.get('contrast', (0.9, 1.1))
            saturation = self.opt.get('saturation', (0.9, 1.1))
            hue = self.opt.get('hue', None)
            natural_real_hq_final = self.color_jitter_pt(img2tensor(natural_real_hq_final, bgr2rgb=True, float32=False), brightness, contrast, saturation, hue)  #RGB Tensor 0~1 C*H*W
            natural_real_hq_final = natural_real_hq_final.numpy().transpose(1,2,0)[:,:,::-1] #transfer back to numpy for the following degradation, 0~1, BGR, H*W*C

        if random.random() > 0.95: #gray
            aug_gray = ia.Sometimes(1, ia.Grayscale(alpha=random.choice([0.8, 0.9, 1.0])))
            natural_real_hq_final = aug_gray(image=np.uint8(natural_real_hq_final*255))/255.0

        if random.random() > 0.8: #same degradation input BGR 0~1, output BGR 0~1
            b1, b2 = self.AddBlur_Two(natural_real_hq_final*255.0, face_real_hq_final*255.0)
            d1, d2 = self.AddDownSample_Two(b1, b2)
            n1, n2 = self.AddNoise_Two(d1, d2)
            j1, j2 = self.AddJPEG_Two(n1, n2)
            natural_syn_lq, face_syn_lq = self.AddUpSample_Two(j1, j2)
            natural_syn_lq_final, face_syn_lq_final = natural_syn_lq/255.0, face_syn_lq/255.0 
        else: #input BGR 0~1 output BGR 0~1
            s1, s2 = same_degradation.degradation_pipeline(natural_real_hq_final*255.0, face_real_hq_final*255.0)
            natural_syn_lq_final, face_syn_lq_final = self.AddUpSample_Two(np.array(s1)/255.0, np.array(s2)/255.0)

        

        
        #########################################################################################
        ########### Pairs 3: synthetic natural pairs for f2n-esrgan
        #########################################################################################
        natural_path = self.df2kpaths[random.randint(0, len(self.df2kpaths)-1)]
        retry = 3
        while retry > 0:
            try:
                hq_bytes = self.file_client.get(natural_path)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                natural_path = self.df2kpaths[random.randint(0, len(self.df2kpaths)-1)]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        try:
            natural_real_hq_restore = imfrombytes(hq_bytes, float32=True)#0~1 BGR
        except:
            print('error in '+reallq_path)

        
        # resize 256 ~ and crop
        random_scale = random.randint(self.out_size, min(natural_real_hq_restore.shape[:2])) / min(natural_real_hq_restore.shape[:2])
        natural_real_hq_restore = cv2.resize(natural_real_hq_restore, (int(random_scale * natural_real_hq_restore.shape[1]), int(random_scale * natural_real_hq_restore.shape[0])), random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
        h2, w2 = natural_real_hq_restore.shape[:2]
        h2_offset = random.randint(0, h2 - self.out_size)
        w2_offset = random.randint(0, w2 - self.out_size)
        natural_real_hq_restore = natural_real_hq_restore[h2_offset:h2_offset+self.out_size, w2_offset:w2_offset+self.out_size, :]

        if random.random() > 0.95: #gray
            aug_gray = ia.Sometimes(1, ia.Grayscale(alpha=random.choice([0.8, 0.9, 1.0])))
            natural_real_hq_restore = aug_gray(image=np.uint8(natural_real_hq_restore*255))/255.0

        if random.random() > 0.6:#real-esrgan
            ##input should be BGR 0~1 numpy H*W*C
            ##output is RGB 0~1 tensor
            natural_syn_lq_restore = real_esrgan_degradation(natural_real_hq_restore, insf=random.choice([1,2,3,4,5])).squeeze(0).detach().numpy() #output numpy c*h*w 0~1 RGB
            natural_syn_lq_restore = natural_syn_lq_restore.transpose((1,2,0)) #transfer to h*w*c
        else:
            ##input should be RGB 0~1 numpy H*W*C
            ##output is RGB 0~1 numpy H*W*C
            gt_tmp = natural_real_hq_restore[:,:,::-1]#transfer to RGB
            natural_syn_lq_restore, _ = bsrgan_degradation(gt_tmp, sf=random.choice([1,2,3,4,5]), lq_patchsize=None)#RGB 0~1 numpy h*w*c
        
        natural_syn_lq_restore = cv2.resize(natural_syn_lq_restore, (self.out_size//4, self.out_size//4), random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
        natural_syn_lq_restore = natural_syn_lq_restore[:,:,::-1].astype(np.float32) # transfer back to BGR
        natural_real_sq_restore = self.usm_sharp(natural_real_hq_restore)


        # cv2.imwrite('face_real_hq_final.png', face_real_hq_final*255)
        # cv2.imwrite('face_real_lq_final.png', face_real_lq_final*255)
        # cv2.imwrite('face_syn_lq_final.png', face_syn_lq_final*255)
        # cv2.imwrite('natural_syn_lq_final.png', natural_syn_lq_final*255)
        # cv2.imwrite('natural_real_hq_final.png', natural_real_hq_final*255)
        # cv2.imwrite('natural_syn_lq_restore.png', natural_syn_lq_restore*255)
        # cv2.imwrite('natural_real_hq_restore.png', natural_real_hq_restore*255)
        # cv2.imwrite('natural_real_sq_restore.png', natural_real_sq_restore*255)
        # exit('check each types of images')
        
        #########################################################################################
        ########### Norm: tensor -1~1 RGB
        #########################################################################################
        face_real_hq_final = img2tensor(face_real_hq_final, bgr2rgb=True, float32=False)
        face_real_lq_final = img2tensor(face_real_lq_final, bgr2rgb=True, float32=False)
    
        face_syn_lq_final = img2tensor(face_syn_lq_final, bgr2rgb=True, float32=False)
        natural_syn_lq_final = img2tensor(natural_syn_lq_final, bgr2rgb=True, float32=False)
        natural_real_hq_final = img2tensor(natural_real_hq_final, bgr2rgb=True, float32=False)

        natural_syn_lq_restore = img2tensor(natural_syn_lq_restore, bgr2rgb=True, float32=False)
        natural_real_hq_restore = img2tensor(natural_real_hq_restore, bgr2rgb=True, float32=False)
        natural_real_sq_restore = img2tensor(natural_real_sq_restore, bgr2rgb=True, float32=False)

        normalize(face_real_hq_final, self.mean, self.std, inplace=True)#-1~1 RGB
        normalize(face_real_lq_final, self.mean, self.std, inplace=True)#-1~1 RGB

        normalize(face_syn_lq_final, self.mean, self.std, inplace=True)#-1~1 RGB
        normalize(natural_syn_lq_final, self.mean, self.std, inplace=True)#-1~1 RGB
        normalize(natural_real_hq_final, self.mean, self.std, inplace=True)#-1~1 RGB

        normalize(natural_syn_lq_restore, self.mean, self.std, inplace=True)#-1~1 RGB
        normalize(natural_real_hq_restore, self.mean, self.std, inplace=True)#-1~1 RGB
        normalize(natural_real_sq_restore, self.mean, self.std, inplace=True)#-1~1 RGB

        return {
            'face_real_hq_final': face_real_hq_final, 'face_real_lq_final':face_real_lq_final, \
            'face_syn_lq_final':face_syn_lq_final, 'natural_syn_lq_final': natural_syn_lq_final, 'natural_real_hq_final': natural_real_hq_final, \
            'natural_syn_lq_restore':natural_syn_lq_restore, 'natural_real_hq_restore': natural_real_hq_restore, 'natural_real_sq_restore': natural_real_sq_restore, 
            }

    def __len__(self):
        return len(self.paths)
