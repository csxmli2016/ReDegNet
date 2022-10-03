
import torch
from utils import utils_image as util
from models.F2NESRGAN import F2NESRGAN
import cv2 
import numpy as np
import os
import time



def main():
    #################################################
    ############### parameter settings ##############
    #################################################
    testsets = './testsets/'#
    testset_Ls = ['lqs']#['whole', 'blurry_faces'] # set path of each sub-set
    model_path = '../experiments/weights/net_f2n_init.pth' #this model is specifically fine-tuned for Figure 1

    t = time.strftime("%m-%d_%H-%M", time.localtime()) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BGModel = F2NESRGAN(CheckPointPath=model_path, device=device)
    
    print('{:>16s} : {:s}'.format('Model Name', 'F2N-ESRGAN'))
    # torch.cuda.set_device(0)      # set GPU ID
    if device == 'cpu':
        print('{:>16s} : {:s}'.format('Using Device', 'CPU'))
    else:
        print('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
    
    torch.cuda.empty_cache()

    for testset_L in testset_Ls:
        print('################################## Handling {:s} ##################################'.format(testset_L))
        L_path = os.path.join(testsets, testset_L)
        Save_path = os.path.join(testsets, testset_L+'_'+t) # save path
        util.mkdir(Save_path)
        print('{:>16s} : {:s}'.format('Input Path', L_path))
        print('{:>16s} : {:s}'.format('Output Path', Save_path))
        idx = 0
        TotalTime = 0
        for img in util.get_image_paths(L_path):
            ####################################
            #####(1) Read Image
            ####################################
            idx += 1
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
    main()