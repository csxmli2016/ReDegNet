import torch
from utils import utils_image as util
from models.F2NESRGAN import F2NESRGAN
import cv2 
import numpy as np
import os
import argparse


def main(args):
    #################################################
    ############### parameter settings ##############
    #################################################
    testsets = './testsets/'
    testset_L = args.input
    model_path = '../experiments/weights/net_f2n_g.pth' 
    '''
    net_f2n_init.pth is specifically finetuned with degradation from Figure 1. So it performs better on Figure 1 but may have obvious artifacts on other old images.
    net_f2n_g.pth is the stable version that is not obviously overfitted to the degradation in Figure 1. (Preffered)
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BGModel = F2NESRGAN(CheckPointPath=model_path, device=device)
    
    print('{:>16s} : {:s}'.format('Model Name', 'F2N-ESRGAN'))
    
    if device == 'cpu':
        print('{:>16s} : {:s}'.format('Using Device', 'CPU'))
    else:
        print('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
    
    torch.cuda.empty_cache()

    print('################################## Handling {:s} ##################################'.format(testset_L))
    L_path = os.path.join(testsets, testset_L)
    Save_path = os.path.join(testsets, testset_L+'_Results') # save path
    util.mkdir(Save_path)
    print('{:>16s} : {:s}'.format('Input Path', L_path))
    print('{:>16s} : {:s}'.format('Output Path', Save_path))
    for img in util.get_image_paths(L_path):
        ####################################
        #####(1) Read Image
        ####################################
        img_name, ext = os.path.splitext(os.path.basename(img))
        print('Restoring {}'.format(img_name))
        img_L = cv2.imread(img)  # 
        img_L = np.ascontiguousarray(img_L[:,:,::-1])

        ####################################
        #####(2) Restoration
        ####################################
        try:
            results = BGModel.handle_restoration(bg=img_L, tile_size=None)
        except:
            print('Using tile operation')
            results = BGModel.handle_restoration(bg=img_L, tile_size=512)
            
        ####################################
        #####(3) Save Results
        ####################################
        util.imsave(results, os.path.join(Save_path, img_name+'_F2NESRGAN.png'))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='lqs', help='Input image or folder')
    args = parser.parse_args()
    main(args)
