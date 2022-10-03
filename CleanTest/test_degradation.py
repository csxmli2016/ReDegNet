
import torch
from utils import utils_image as util
import cv2 
import numpy as np
import os.path
import time
from models import networks
from PIL import Image
import os
import random
import torchvision.transforms as transforms


def main():
    TIMESTAMP = time.strftime("%m-%d_%H-%M", time.localtime()) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = './SynDegResults'
    os.makedirs(save_path, exist_ok=True)

    modelsynE = networks.SynNetEncoder()
    modelsynG = networks.SynNetGenerator()
    modelsynDeg = networks.DegNet()

    modelsynE.load_state_dict(torch.load('../experiments/weights/net_synE_init.pth')['params'], strict=True)
    modelsynG.load_state_dict(torch.load('../experiments/weights/net_synG_init.pth')['params'], strict=True)
    modelsynDeg.load_state_dict(torch.load('../experiments/weights/net_deg_init.pth')['params'], strict=True)
    modelsynE.eval()
    modelsynG.eval()
    modelsynDeg.eval()
    for k, v in modelsynE.named_parameters():
        v.requires_grad = False
    for k, v in modelsynG.named_parameters():
        v.requires_grad = False
    for k, v in modelsynDeg.named_parameters():
        v.requires_grad = False
    
    modelsynE = modelsynE.to(device)
    modelsynG = modelsynG.to(device)
    modelsymDeg = modelsynDeg.to(device)
    torch.cuda.empty_cache()
    
    RealFaceLQ = Image.open('../experiments/validation/FaceLQ/000013_face00_scale5.419.png').convert('RGB')# Take Figure 1 for example
    RealFaceHQ = Image.open('../experiments/validation/FaceHQ/000013_face00_scale5.419.png').convert('RGB')# Take Figure 1 for example
    NaturalHQ = Image.open('../TrainData/DF2K_HQPatches/000446_00010_10.png').convert('RGB') #reading the HQ natural image
    '''
    Note: the scale here referes to the upsample scale factor that the original LQ face resized to HQ size (i.e., 512)
    You can use any robust blind face restoration methods, like GPEN, and GFPGAN.
    Please refer to our synthetic process in Line144~171 of redeg_model.py
    '''
    x = 120 
    y = 180
    l = 256
    RealFaceLQ = RealFaceLQ.crop((x,y,x+l,y+l)) #remove background space
    RealFaceHQ = RealFaceHQ.crop((x,y,x+l,y+l)) #remove background space

    x = random.randint(0,100) # random crop from 400*400 natural HQ patches
    y = random.randint(0,100) # random crop from 400*400 natural HQ patches
    NaturalHQ = NaturalHQ.crop((x, y, x+256, y+256))

    RealFaceLQ = transforms.ToTensor()(RealFaceLQ)
    RealFaceLQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(RealFaceLQ.unsqueeze(0)).to(device)
    RealFaceHQ = transforms.ToTensor()(RealFaceHQ)
    RealFaceHQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(RealFaceHQ.unsqueeze(0)).to(device)
    RealNaturalHQ = transforms.ToTensor()(NaturalHQ)
    RealNaturalHQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(RealNaturalHQ.unsqueeze(0)).to(device)

    '''
    Step 1: Obtain the degradation from face pairs
    '''
    RealFaceHLQ = torch.cat([RealFaceHQ, RealFaceLQ], dim=1)
    RealDeg = modelsynDeg(RealFaceHLQ) #this the degradatoin representation
    print('Step 1: Obtain the degradation from face pairs')

    '''
    Step 2: Obtain the HQ image features 
    '''
    CFace = modelsynE(RealFaceHQ)
    CNatural = modelsynE(RealNaturalHQ)
    print('Step 2: Obtain the HQ image features ')

    '''
    Step 3: Synthesize the LQ image with the degradation from extracted degradation representation from Step 1 
    '''
    SynLQFace = modelsynG(styles=[RealDeg], noise=CFace)
    SynLQNatural = modelsynG(styles=[RealDeg], noise=CNatural)
    print('Step 3: Synthesize the LQ image with the degradation from extracted degradation representation from Step 1')

    '''
    Step 4: Save results
    '''
    SynLQFace = SynLQFace * 0.5 + 0.5
    SynLQFace = SynLQFace.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
    SynLQFace = np.clip(SynLQFace.float().cpu().numpy(), 0, 1)[:,:,::-1] * 255.0
    util.imsave(SynLQFace, os.path.join(save_path, 'Face_SynDeg.png'))

    SynLQNatural = SynLQNatural * 0.5 + 0.5
    SynLQNatural = SynLQNatural.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
    SynLQNatural = np.clip(SynLQNatural.float().cpu().numpy(), 0, 1)[:,:,::-1] * 255.0
    util.imsave(SynLQNatural, os.path.join(save_path, 'Natural_SynDeg.png'))

    print('Saving the synthetic images...')





if __name__ == '__main__':
    main()
