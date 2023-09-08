import pandas as pd
import numpy as np

from PIL import Image
import os
import scipy
import cv2

from tqdm import tqdm
from time import time
import json 

with open('config.json') as f_in:
    config = json.load(f_in)


def std_convoluted(im, N):
    """
    Calculate the standard deviation using a convolution operation.

    Parameters:
    - im (numpy.ndarray): Input image as a NumPy array.
    - N (int): Radius of the convolution kernel.

    Returns:
    - numpy.ndarray: Standard deviation image.
    """
    # Calculate squared image and initialize kernel
    im2 = im**2
    ones = np.ones(im.shape)
    kernel = np.ones((2*N+1, 2*N+1))

    # Convolve the image and squared image with the kernel
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")

    # Calculate the standard deviation image
    return np.sqrt((s2 - s**2 / ns) / ns)

def my_func(path_to_orthophoto_rgb):
    
    Image.MAX_IMAGE_PIXELS = None
    
    rgb_name = os.path.basename(path_to_orthophoto_rgb)
    dept = rgb_name[:2]
    year = rgb_name[3:7]
    path_to_orthophoto_irc = config["irc_path"][dept][year] + rgb_name[:config['irc_pos']] + '-IRC' + rgb_name[config['irc_pos']:]# config["irc_path"] + path_to_orthophoto_rgb[config['rgb_name_pos']:] # A Changer plus tard
    # mnhc_tiff = config['mnhc_path'] + "Diff_MNS_CORREL_1-0_LAMB93_20FD3525_" + str(path_to_orthophoto_rgb[config['rgb_coordinates_pos']:config['rgb_coordinates_pos']+8]).replace('-', '_') + ".tif"
    
    # position
    rgb_x = path_to_orthophoto_rgb[config['rgb_coordinates_pos']: config['rgb_coordinates_pos']+3]
    rgb_y = path_to_orthophoto_rgb[config['rgb_coordinates_pos']+4: config['rgb_coordinates_pos']+8]
    mnhc_dir_dept = config['mnhc_path']+dept+'_'+year+'/'
    
    # images IRC et RGB
    # ortho_irc_file = config["irc_path"] + '/' + dept + '-2020-0'+rgb_x+'-'+rgb_y+'-LA93-0M20-IRC-E080.jp2'
    # ouverture
    ortho_rgb = cv2.resize(np.asarray(Image.open(path_to_orthophoto_rgb)), (10000, 10000), interpolation=cv2.INTER_AREA)
    ortho_irc = cv2.resize(np.asarray(Image.open(path_to_orthophoto_irc)), (10000, 10000), interpolation=cv2.INTER_AREA) 
    ndvi = np.divide(ortho_irc[:,:,0]-ortho_irc[:,:,1],ortho_irc[:,:,0]+ortho_irc[:,:,1], where=(ortho_irc[:,:,0]+ortho_irc[:,:,1])!=0 )
    ortho_irc = None 
    
    if not os.path.exists(config['rgb_vignettes_path'] + f'rgb_{rgb_x}_{rgb_y}/'):
        os.makedirs(config['rgb_vignettes_path'] + f'rgb_{rgb_x}_{rgb_y}/')
    if not os.path.exists(config['mask_vignettes_path'] + f'mask_{rgb_x}_{rgb_y}/'):
        os.makedirs(config['mask_vignettes_path'] + f'mask_{rgb_x}_{rgb_y}/')

    # i = 0
    vignette_data = []
    for mnhc_x in tqdm(range(0,5)):
        for mnhc_y in range(0,5):
            mnhc_file = mnhc_dir_dept+'Diff_MNS_CORREL_1-0_LAMB93_20FD'+dept+'25_'+str(int(int(rgb_x)+mnhc_x))+'_'+str(int(int(rgb_y)-mnhc_y))+'.tif'
            
            if os.path.exists(mnhc_file):
                # print(mnhc_file)
                mnhc = np.asarray(Image.open(mnhc_file))
                w, h = mnhc.shape
                mask = np.logical_and(np.logical_and(mnhc > 3., 
                                                     np.logical_or(mnhc > 5., 
                                                                   std_convoluted(mnhc,N=10) > .5)
                                                    ),
                                      ndvi[mnhc_x*w:(mnhc_x+1)*w, mnhc_y*h:(mnhc_y+1)*h] > 0) \
                .astype('uint8')

                
                for x in range(0, w-config['img_size']+1, config['img_size']):
                    for y in range(0, h-config['img_size']+1, config['img_size']):
                        # print(x, y)
                        crop_rgb = ortho_rgb[mnhc_y*h+y: mnhc_y*h+y+config['img_size'], mnhc_x*w+x: mnhc_x*w+x+config['img_size'], :]
                        crop_mask = mask[y:y+config['img_size'], x:x+config['img_size']]

                        # filtre rgb
                        # enregistrement
                        pos = dept+'_'+str(int((int(rgb_x)+mnhc_x)*1000+x/2))+'_'+str(int((int(rgb_y)-mnhc_y)*1000+y/2))
                        
                        cv2.imwrite(config['rgb_vignettes_path'] + f'rgb_{rgb_x}_{rgb_y}/' + 'rgb_' + str(pos) + '.jpg', crop_rgb)
                        cv2.imwrite(config['mask_vignettes_path'] + f'mask_{rgb_x}_{rgb_y}/' + 'mask_'+ str(pos) + '.png', crop_mask)

                        # Calculate characteristics of the RGB crop and add to vignette_data
                        mask_2 = np.stack((crop_mask,)*3, axis=-1)
                        non_mask = np.logical_not(mask_2)
                        res = [pos, crop_mask.sum()] + \
                        np.mean(crop_rgb, axis=(0, 1), where=mask_2.astype(bool)).tolist() + \
                        np.mean(crop_rgb, axis=(0, 1), where=non_mask.astype(bool)).tolist() + \
                        np.std(crop_rgb, axis=(0, 1), where=mask_2.astype(bool)).tolist() + \
                        np.std(crop_rgb, axis=(0, 1), where=non_mask.astype(bool)).tolist()
                        vignette_data.append(res)
                
                # Si besoin de plus d'images, commenter le break
                break
    
    # Convert the vignette data to a Pandas DataFrame and return
    return pd.DataFrame(vignette_data, columns=('file', 'wood_size', 'w_mean_R', 'w_mean_G', 'w_mean_B', 'nw_mean_R', 'nw_mean_G', 'nw_mean_B', 'w_std_R', 'w_std_G', 'w_std_B', 'nw_std_R', 'nw_std_G', 'nw_std_B'))
                