import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
import skimage.transform as st

from PIL import Image
import os
import scipy
import cv2

from tqdm import tqdm
from time import time
import json 

import warnings
warnings.simplefilter("ignore")

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

def convolution(im, N):
    """
    Apply a simple convolution operation to an input image 
    using a square kernel of size (2N+1)x(2N+1).

    Args:
    im (numpy.ndarray): The input image as a numpy array.
    N (int): The size parameter used to define the kernel.

    Returns:
    numpy.ndarray: The convolved image.
    """
    # Calculate squared image and initialize kernel
    kernel = np.ones((2*N+1, 2*N+1))
    # Convolve the image and squared image with the kernel
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    return s


def vignette_and_mask_creation_func(path_to_orthophoto_rgb, gdf):
    """
    Create and save vignettes and masks from the RGB and IRC orthophotos.

    Parameters:
    - path_to_orthophoto_rgb (str): Path to the orthophoto RGB image.

    Returns:
    - pd.DataFrame: DataFrame containing stats about the vignettes.
    """
    Image.MAX_IMAGE_PIXELS = None
    
    rgb_name = os.path.basename(path_to_orthophoto_rgb)
    dept = rgb_name[:2]
    year = rgb_name[3:7]
    path_to_orthophoto_irc = config["irc_path"][dept][year] + rgb_name[:config['irc_pos']] + '-IRC' + rgb_name[config['irc_pos']:]

    # print(dept, year, rgb_name, path_to_orthophoto_irc)
    
    # Extract coordinates from RGB orthophoto path
    rgb_x = path_to_orthophoto_rgb[config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']: config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+3]
    rgb_y = path_to_orthophoto_rgb[config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+4: config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+8]
    mnhc_dir_dept = config['mnhc_path']+dept+'_'+year+'/'
    
    # Open and resize IRC and RGB images
    ortho_rgb = np.asarray(Image.open(path_to_orthophoto_rgb))
    ortho_irc = np.asarray(Image.open(path_to_orthophoto_irc))
    
    with rasterio.open(path_to_orthophoto_rgb) as src:
        # aerial_image = src.read()
        aerial_transform = src.transform
    
    # Create a mask with the same dimensions as the orthophoto
    buildings = geometry_mask(gdf.geometry, transform=aerial_transform, invert=True, out_shape=ortho_rgb.shape[:-1])
    
    ortho_rgb = cv2.resize(ortho_rgb, (10000, 10000), interpolation=cv2.INTER_AREA)
    ortho_irc = cv2.resize(ortho_irc, (10000, 10000), interpolation=cv2.INTER_AREA) 
    ndvi = np.divide(ortho_irc[:,:,0]-ortho_irc[:,:,1],ortho_irc[:,:,0]+ortho_irc[:,:,1], where=(ortho_irc[:,:,0]+ortho_irc[:,:,1])!=0)
    ortho_irc = None 
    buildings = st.resize(buildings.astype(int), (10000, 10000), order=0, preserve_range=True, anti_aliasing=False)
    

    # Create output directories if they don't exist
    if not os.path.exists(config['rgb_vignettes_path'] + f'{dept}/' + f'rgb_{rgb_x}_{rgb_y}/'):
        os.makedirs(config['rgb_vignettes_path'] + f'{dept}/' + f'rgb_{rgb_x}_{rgb_y}/')
    if not os.path.exists(config['mask_vignettes_path'] + f'{dept}/' + f'mask_{rgb_x}_{rgb_y}/'):
        os.makedirs(config['mask_vignettes_path'] + f'{dept}/' + f'mask_{rgb_x}_{rgb_y}/')

    # Read existing DataFrame or create a new one
    if os.path.exists(config['stat_vignettes_path']):
        df = pd.read_csv(config['stat_vignettes_path'], index_col="Unnamed: 0")
    else:
        df = pd.DataFrame(columns=config['stat_vignettes_col'])

    for mnhc_x in tqdm(range(0,5)):
        for mnhc_y in range(0,5):
            if dept == '35':
                mnhc_file = mnhc_dir_dept+'Diff_MNS_CORREL_1-0_LAMB93_20FD'+dept+'25_'+str(int(int(rgb_x)+mnhc_x))+'_'+str(int(int(rgb_y)-mnhc_y))+'.tif'
            else:
                mnhc_file = mnhc_dir_dept+'Diff_MNS_CORREL_1-0_LAMB93_20FD'+dept+'25_'+str(int(int(rgb_x)+mnhc_x))+'_'+str(int(int(rgb_y)-(mnhc_y+1)))+'.tif'
            
            if os.path.exists(mnhc_file):
                # print(mnhc_file)
                mnhc = np.asarray(Image.open(mnhc_file))
                w, h = mnhc.shape
                std_conv_mnhc = std_convoluted(mnhc,N=10)
                if dept == '35':
                    mask = np.logical_and(np.logical_and(mnhc > 3., 
                                                         np.logical_or(mnhc > 5., 
                                                                       std_conv_mnhc > .5)
                                                        ),
                                          np.logical_and(ndvi[mnhc_x*w:(mnhc_x+1)*w, mnhc_y*h:(mnhc_y+1)*h] > 0,
                                                         buildings[mnhc_x*w:(mnhc_x+1)*w, mnhc_y*h:(mnhc_y+1)*h] == 0
                                                        ) \
                                         ).astype('uint8')
                else:
                    mask = np.logical_and(np.logical_and(mnhc > 3., 
                                                         np.logical_or(mnhc > 5., 
                                                                       std_conv_mnhc > .5)
                                                        ),
                                          np.logical_and(ndvi[mnhc_y*h:(mnhc_y+1)*h, mnhc_x*w:(mnhc_x+1)*w] > 0,
                                                         buildings[mnhc_y*h:(mnhc_y+1)*h, mnhc_x*w:(mnhc_x+1)*w] == 0
                                                        ) \
                                         ).astype('uint8')
                # mnhc_y*h:(mnhc_y+1)*h, mnhc_x*w:(mnhc_x+1)*w
                
                for x in range(0, w-(config['img_size']+config['border']*2)+1, config['img_size']+config['border']*2):
                    for y in range(0, h-(config['img_size']+config['border']*2)+1, config['img_size']+config['border']*2):
                        
                        crop_rgb = ortho_rgb[mnhc_y*h+y: mnhc_y*h+y+config['img_size']+config['border']*2, mnhc_x*w+x: mnhc_x*w+x+config['img_size']+config['border']*2, :]
                        crop_mask = mask[y:y+config['img_size']+config['border']*2, x:x+config['img_size']+config['border']*2]

                        pos = dept+'_'+str(int((int(rgb_x)+mnhc_x)*1000+x/2))+'_'+str(int((int(rgb_y)-mnhc_y)*1000+y/2))
                        
                        cv2.imwrite(config['rgb_vignettes_path'] + f'{dept}/' + f'rgb_{rgb_x}_{rgb_y}/' + 'rgb_' + str(pos) + '.jpg', crop_rgb)
                        cv2.imwrite(config['mask_vignettes_path'] + f'{dept}/' + f'mask_{rgb_x}_{rgb_y}/' + 'mask_'+ str(pos) + '.png', crop_mask*255)

                        # Calculate characteristics of the RGB crop and add to the DataFrame
                        mask_2 = np.stack((crop_mask,)*3, axis=-1)
                        non_mask = np.logical_not(mask_2)
                        non_crop_mask = np.logical_not(crop_mask)
                        res = [dept, pos, crop_mask.sum()] + \
                        np.mean(crop_rgb, axis=(0, 1), where=mask_2.astype(bool)).tolist() + \
                        np.mean(crop_rgb, axis=(0, 1), where=non_mask.astype(bool)).tolist() + \
                        np.std(crop_rgb, axis=(0, 1), where=mask_2.astype(bool)).tolist() + \
                        np.std(crop_rgb, axis=(0, 1), where=non_mask.astype(bool)).tolist() + \
                        [np.mean(convolution(crop_rgb[:, :, 0], N=3), where=crop_mask.astype(bool)), np.mean(convolution(crop_rgb[:, :, 1], N=3), where=crop_mask.astype(bool)), np.mean(convolution(crop_rgb[:, :, 2], N=3), where=crop_mask.astype(bool))] + \
                        [np.mean(convolution(crop_rgb[:, :, 0], N=3), where=non_crop_mask.astype(bool)), np.mean(convolution(crop_rgb[:, :, 1], N=3), where=non_crop_mask.astype(bool)), np.mean(convolution(crop_rgb[:, :, 2], N=3), where=non_crop_mask.astype(bool))] + \
                        [np.std(convolution(crop_rgb[:, :, 0], N=3), where=crop_mask.astype(bool)), np.std(convolution(crop_rgb[:, :, 1], N=3), where=crop_mask.astype(bool)), np.std(convolution(crop_rgb[:, :, 2], N=3), where=crop_mask.astype(bool))] + \
                        [np.std(convolution(crop_rgb[:, :, 0], N=3), where=non_crop_mask.astype(bool)), np.std(convolution(crop_rgb[:, :, 1], N=3), where=non_crop_mask.astype(bool)), np.std(convolution(crop_rgb[:, :, 2], N=3), where=non_crop_mask.astype(bool))] 
                        # vignette_data.append(res)
                        
                        # Append the list to the DataFrame
                        df.loc[len(df) + len(set(range(df.index[-1]))-set(df.index))
                        if len(df.index) > 0
                        else 0] = res
                
                # If more images are needed, comment the break
                break
                
    # Drop duplicates and save the DataFrame to a CSV file
    df.drop_duplicates(subset='file').to_csv(config['stat_vignettes_path'])
    
    return df

def large_vignette_and_mask_creation_func(path_to_orthophoto_rgb, gdf):
    """
    Create and save vignettes and masks from the RGB and IRC orthophotos.

    Parameters:
    - path_to_orthophoto_rgb (str): Path to the orthophoto RGB image.

    Returns:
    - pd.DataFrame: DataFrame containing stats about the vignettes.
    """
    Image.MAX_IMAGE_PIXELS = None
    
    rgb_name = os.path.basename(path_to_orthophoto_rgb)
    dept = rgb_name[:2]
    year = rgb_name[3:7]
    path_to_orthophoto_irc = config["irc_path"][dept][year] + rgb_name[:config['irc_pos']] + '-IRC' + rgb_name[config['irc_pos']:]

    # print(dept, year, rgb_name, path_to_orthophoto_irc)
    
    # Extract coordinates from RGB orthophoto path
    rgb_x = path_to_orthophoto_rgb[config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']: config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+3]
    rgb_y = path_to_orthophoto_rgb[config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+4: config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+8]
    mnhc_dir_dept = config['mnhc_path']+dept+'_'+year+'/'
    
    # Open and resize IRC and RGB images
    ortho_rgb = np.asarray(Image.open(path_to_orthophoto_rgb))
    ortho_irc = np.asarray(Image.open(path_to_orthophoto_irc))
    
    with rasterio.open(path_to_orthophoto_rgb) as src:
        # aerial_image = src.read()
        aerial_transform = src.transform
    
    # Create a mask with the same dimensions as the orthophoto
    buildings = geometry_mask(gdf.geometry, transform=aerial_transform, invert=True, out_shape=ortho_rgb.shape[:-1])
    
    ortho_rgb = cv2.resize(ortho_rgb, (10000, 10000), interpolation=cv2.INTER_AREA)
    ortho_irc = cv2.resize(ortho_irc, (10000, 10000), interpolation=cv2.INTER_AREA) 
    ndvi = np.divide(ortho_irc[:,:,0]-ortho_irc[:,:,1],ortho_irc[:,:,0]+ortho_irc[:,:,1], where=(ortho_irc[:,:,0]+ortho_irc[:,:,1])!=0)
    ortho_irc = None 
    buildings = st.resize(buildings.astype(int), (10000, 10000), order=0, preserve_range=True, anti_aliasing=False)
    

    # Create output directories if they don't exist
    if not os.path.exists(config['rgb_vignettes_path'] + f'{dept}_large/' + f'rgb_{rgb_x}_{rgb_y}/'):
        os.makedirs(config['rgb_vignettes_path'] + f'{dept}_large/' + f'rgb_{rgb_x}_{rgb_y}/')
    if not os.path.exists(config['mask_vignettes_path'] + f'{dept}_large/' + f'mask_{rgb_x}_{rgb_y}/'):
        os.makedirs(config['mask_vignettes_path'] + f'{dept}_large/' + f'mask_{rgb_x}_{rgb_y}/')

    # Read existing DataFrame or create a new one
    # if os.path.exists(config['stat_vignettes_path']):
    #     df = pd.read_csv(config['stat_vignettes_path'], index_col="Unnamed: 0")
    # else:
    #     df = pd.DataFrame(columns=config['stat_vignettes_col'])

    for mnhc_x in tqdm(range(0,5)):
        for mnhc_y in range(0,5):
            if dept == '35':
                mnhc_file = mnhc_dir_dept+'Diff_MNS_CORREL_1-0_LAMB93_20FD'+dept+'25_'+str(int(int(rgb_x)+mnhc_x))+'_'+str(int(int(rgb_y)-mnhc_y))+'.tif'
            else:
                mnhc_file = mnhc_dir_dept+'Diff_MNS_CORREL_1-0_LAMB93_20FD'+dept+'25_'+str(int(int(rgb_x)+mnhc_x))+'_'+str(int(int(rgb_y)-(mnhc_y+1)))+'.tif'
            
            if os.path.exists(mnhc_file):
                # print(mnhc_file)
                mnhc = np.asarray(Image.open(mnhc_file))
                w, h = mnhc.shape
                std_conv_mnhc = std_convoluted(mnhc,N=10)
                if dept == '35':
                    mask = np.logical_and(np.logical_and(mnhc > 3., 
                                                         np.logical_or(mnhc > 5., 
                                                                       std_conv_mnhc > .5)
                                                        ),
                                          np.logical_and(ndvi[mnhc_x*w:(mnhc_x+1)*w, mnhc_y*h:(mnhc_y+1)*h] > 0,
                                                         buildings[mnhc_x*w:(mnhc_x+1)*w, mnhc_y*h:(mnhc_y+1)*h] == 0
                                                        ) \
                                         ).astype('uint8')
                else:
                    mask = np.logical_and(np.logical_and(mnhc > 3., 
                                                         np.logical_or(mnhc > 5., 
                                                                       std_conv_mnhc > .5)
                                                        ),
                                          np.logical_and(ndvi[mnhc_y*h:(mnhc_y+1)*h, mnhc_x*w:(mnhc_x+1)*w] > 0,
                                                         buildings[mnhc_y*h:(mnhc_y+1)*h, mnhc_x*w:(mnhc_x+1)*w] == 0
                                                        ) \
                                         ).astype('uint8')
                # mnhc_y*h:(mnhc_y+1)*h, mnhc_x*w:(mnhc_x+1)*w
                
                for x in range(0, w-(config['large_img_size']+config['border']*2)+1, config['large_img_size']+config['border']*2):
                    for y in range(0, h-(config['large_img_size']+config['border']*2)+1, config['large_img_size']+config['border']*2):
                        
                        crop_rgb = ortho_rgb[mnhc_y*h+y: mnhc_y*h+y+config['large_img_size']+config['border']*2, mnhc_x*w+x: mnhc_x*w+x+config['large_img_size']+config['border']*2, :]
                        crop_mask = mask[y:y+config['large_img_size']+config['border']*2, x:x+config['large_img_size']+config['border']*2]

                        pos = dept+'_'+str(int((int(rgb_x)+mnhc_x)*1000+x/2))+'_'+str(int((int(rgb_y)-mnhc_y)*1000+y/2))
                        
                        cv2.imwrite(config['rgb_vignettes_path'] + f'{dept}_large/' + f'rgb_{rgb_x}_{rgb_y}/' + 'rgb_' + str(pos) + '.jpg', crop_rgb)
                        cv2.imwrite(config['mask_vignettes_path'] + f'{dept}_large/' + f'mask_{rgb_x}_{rgb_y}/' + 'mask_'+ str(pos) + '.png', crop_mask*255)

                        # Calculate characteristics of the RGB crop and add to the DataFrame
#                         mask_2 = np.stack((crop_mask,)*3, axis=-1)
#                         non_mask = np.logical_not(mask_2)
#                         non_crop_mask = np.logical_not(crop_mask)
#                         res = [dept, pos, crop_mask.sum()] + \
#                         np.mean(crop_rgb, axis=(0, 1), where=mask_2.astype(bool)).tolist() + \
#                         np.mean(crop_rgb, axis=(0, 1), where=non_mask.astype(bool)).tolist() + \
#                         np.std(crop_rgb, axis=(0, 1), where=mask_2.astype(bool)).tolist() + \
#                         np.std(crop_rgb, axis=(0, 1), where=non_mask.astype(bool)).tolist() + \
#                         [np.mean(convolution(crop_rgb[:, :, 0], N=3), where=crop_mask.astype(bool)), np.mean(convolution(crop_rgb[:, :, 1], N=3), where=crop_mask.astype(bool)), np.mean(convolution(crop_rgb[:, :, 2], N=3), where=crop_mask.astype(bool))] + \
#                         [np.mean(convolution(crop_rgb[:, :, 0], N=3), where=non_crop_mask.astype(bool)), np.mean(convolution(crop_rgb[:, :, 1], N=3), where=non_crop_mask.astype(bool)), np.mean(convolution(crop_rgb[:, :, 2], N=3), where=non_crop_mask.astype(bool))] + \
#                         [np.std(convolution(crop_rgb[:, :, 0], N=3), where=crop_mask.astype(bool)), np.std(convolution(crop_rgb[:, :, 1], N=3), where=crop_mask.astype(bool)), np.std(convolution(crop_rgb[:, :, 2], N=3), where=crop_mask.astype(bool))] + \
#                         [np.std(convolution(crop_rgb[:, :, 0], N=3), where=non_crop_mask.astype(bool)), np.std(convolution(crop_rgb[:, :, 1], N=3), where=non_crop_mask.astype(bool)), np.std(convolution(crop_rgb[:, :, 2], N=3), where=non_crop_mask.astype(bool))] 
#                         # vignette_data.append(res)
                        
#                         # Append the list to the DataFrame
#                         df.loc[len(df) + len(set(range(df.index[-1]))-set(df.index))
#                         if len(df.index) > 0
#                         else 0] = res
                
                # If more images are needed, comment the break
                # break
                
#     # Drop duplicates and save the DataFrame to a CSV file
#     df.drop_duplicates(subset='file').to_csv(config['stat_vignettes_path'])
    
#     return df

def vignette_to_predict_creation(path_to_orthophoto_rgb, create_stats=False):
    """
    Create vignettes from an orthophoto and optionally update statistics.

    Args:
        path_to_orthophoto_rgb (str): Path to the RGB orthophoto.
        create_stats (bool): Whether to create statistics.

    Returns:
        tuple or None: If create_stats is True, returns ortho_positions, ortho_shapes, and df; otherwise, returns None.

    This function processes an RGB orthophoto and creates vignettes from it. It can also update statistics if create_stats is True.

    :param path_to_orthophoto_rgb: Path to the RGB orthophoto.
    :param create_stats: Whether to create statistics.
    :return: If create_stats is True, returns ortho_positions, ortho_shapes, and df; otherwise, returns ortho_positions, ortho_shapes.
    """
    Image.MAX_IMAGE_PIXELS = None

    rgb_name = os.path.basename(path_to_orthophoto_rgb)
    dept = rgb_name[:2]
    year = rgb_name[3:7]

    
    # Extract coordinates from RGB orthophoto path
    rgb_x = path_to_orthophoto_rgb[config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']: config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+3]
    rgb_y = path_to_orthophoto_rgb[config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+4: config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+8]

    # Open and resize RGB images
    ortho_rgb = np.asarray(Image.open(path_to_orthophoto_rgb))
    
    ortho_rgb_shape = ortho_rgb.shape
    
    w = np.min([10000, ortho_rgb_shape[0]])
    h = np.min([10000, ortho_rgb_shape[1]])
    
    # print(h, w)
    
    if ortho_rgb_shape[0] > 10000 or ortho_rgb_shape[1] > 10000:
        ortho_rgb = cv2.resize(ortho_rgb, (10000, 10000), interpolation=cv2.INTER_AREA)
    
    # Create output directories if they don't exist
    if not os.path.exists(config['rgb_older_vignettes_path'] + f'rgb_{year}/'):
        os.makedirs(config['rgb_older_vignettes_path'] + f'rgb_{year}/')
    if not os.path.exists(config['rgb_older_vignettes_path'] + f'rgb_{year}/rgb_{year}_{rgb_x}_{rgb_y}/'):
        os.makedirs(config['rgb_older_vignettes_path'] + f'rgb_{year}/rgb_{year}_{rgb_x}_{rgb_y}/')


    if create_stats:
        # Read existing DataFrame or create a new one
        if os.path.exists(config['stat_older_vignettes_path']):
            df = pd.read_csv(config['stat_older_vignettes_path'], index_col="Unnamed: 0")
        else:
            df = pd.DataFrame(columns=config['stat_older_vignettes_col'])

    if os.path.exists(config['ortho_shapes_path'] + 'ortho_shapes.json'):
        with open(config['ortho_shapes_path'] + 'ortho_shapes.json') as f:
            ortho_shapes = json.load(f)
    else:
        ortho_shapes = {}

    if os.path.exists(config['ortho_positions_path'] + 'ortho_positions.json'):
        with open(config['ortho_positions_path'] + 'ortho_positions.json') as f:
            ortho_positions = json.load(f)
    else:
        ortho_positions = {}

    if os.path.exists(config['ortho_borders_path'] + 'ortho_borders.json'):
        with open(config['ortho_borders_path'] + 'ortho_borders.json') as f:
            ortho_borders = json.load(f)
    else:
        ortho_borders = {}
    
    shapes = {}
    positions = {}
    borders = {}

    for x in range(0, w-(config['img_size'])+1, config['img_size']):
        for y in range(0, h-(config['img_size'])+1, config['img_size']):

            crop_rgb = ortho_rgb[np.max([0, y-config['border']]): y+config['img_size']+config['border'], np.max([0, x-config['border']]): x+config['img_size']+config['border'], :]
            # ortho_rgb[np.max([0, mnhc_y*h+y-config['border']]): mnhc_y*h+y+config['img_size']+config['border'], np.max([0, mnhc_x*w+x-config['border']]): mnhc_x*w+x+config['img_size']+config['border'], :]

            pos = dept+'_'+str(int(int(rgb_x)*1000+x/2))+'_'+str(int(int(rgb_y)*1000+y/2)) 
            #  dept+'_'+str(int((int(rgb_x)+mnhc_x)*1000+x/2))+'_'+str(int((int(rgb_y)-mnhc_y)*1000+y/2))

            cv2.imwrite(config['rgb_older_vignettes_path'] + f'rgb_{year}/rgb_{year}_{rgb_x}_{rgb_y}/' + f'rgb_{year}_' + str(pos) + '.jpg', crop_rgb)

            
            # print({
            #     y-config['border']: True if y-config['border'] > 0 else False,
            #       y+config['img_size']+config['border']: True if y+config['img_size']+config['border'] < h else False,
            #       x-config['border']:True if x-config['border'] > 0 else False,
            #       x+config['img_size']+config['border']: True if x+config['img_size']+config['border'] < w else False
            #     }
            #     )
            
            positions[pos] = tuple(np.array([np.max([0, y-config['border']]), y+config['img_size']+config['border'], np.max([0, x-config['border']]), x+config['img_size']+config['border']])/100)
            shapes[pos] = crop_rgb.shape
            borders[pos] = {'top': True if y-config['border'] > 0 else False,
                            'left': True if x-config['border'] > 0 else False,
                            'right': True if x+config['img_size']+config['border'] < w else False,
                            'bottom': True if y+config['img_size']+config['border'] < h else False}
            
            if create_stats:
            # Calculate characteristics of the RGB crop and add to the DataFrame
                res = [year, pos, crop_rgb.sum()] + \
                    np.mean(crop_rgb, axis=(0, 1)).tolist() + \
                    np.std(crop_rgb, axis=(0, 1)).tolist() + \
                    [np.mean(convolution(crop_rgb[:, :, 0], N=3)), np.mean(convolution(crop_rgb[:, :, 1], N=3)), np.mean(convolution(crop_rgb[:, :, 2], N=3))] + \
                    [np.std(convolution(crop_rgb[:, :, 0], N=3)), np.std(convolution(crop_rgb[:, :, 1], N=3)), np.std(convolution(crop_rgb[:, :, 2], N=3))] 

                # Append the list to the DataFrame
                df.loc[len(df) + len(set(range(df.index[-1]))-set(df.index))
                if len(df.index) > 0
                else 0] = res



    ortho_shapes[rgb_name] = shapes
    with open('vignettes/rgb_older/ortho_shapes.json', 'w') as f:
        json.dump(ortho_shapes, f)

    ortho_positions[rgb_name] = positions
    with open('vignettes/rgb_older/ortho_positions.json', 'w') as f:
        json.dump(ortho_positions, f)
        
    ortho_borders[rgb_name] = borders
    with open('vignettes/rgb_older/ortho_borders.json', 'w') as f:
        json.dump(ortho_borders, f)
    
    if create_stats:
        # Drop duplicates and save the DataFrame to a CSV file
        df.drop_duplicates(subset='file').to_csv(config['stat_older_vignettes_path'])

        return ortho_positions, ortho_shapes, ortho_borders, df
    
    else:
        return ortho_positions, ortho_shapes, ortho_borders
    
    
def large_vignette_to_predict_creation(path_to_orthophoto_rgb, create_stats=False):
    """
    Create vignettes from an orthophoto and optionally update statistics.

    Args:
        path_to_orthophoto_rgb (str): Path to the RGB orthophoto.
        create_stats (bool): Whether to create statistics.

    Returns:
        tuple or None: If create_stats is True, returns ortho_positions, ortho_shapes, and df; otherwise, returns None.

    This function processes an RGB orthophoto and creates vignettes from it. It can also update statistics if create_stats is True.

    :param path_to_orthophoto_rgb: Path to the RGB orthophoto.
    :param create_stats: Whether to create statistics.
    :return: If create_stats is True, returns ortho_positions, ortho_shapes, and df; otherwise, returns ortho_positions, ortho_shapes.
    """
    Image.MAX_IMAGE_PIXELS = None

    rgb_name = os.path.basename(path_to_orthophoto_rgb)
    dept = rgb_name[:2]
    year = rgb_name[3:7]

    
    # Extract coordinates from RGB orthophoto path
    rgb_x = path_to_orthophoto_rgb[config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']: config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+3]
    rgb_y = path_to_orthophoto_rgb[config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+4: config['rgb_coordinates_pos'][str(year) if str(year) == '2006' else 'autre']+8]

    # Open and resize RGB images
    ortho_rgb = np.asarray(Image.open(path_to_orthophoto_rgb))
    
    ortho_rgb_shape = ortho_rgb.shape
    
    w = np.min([10000, ortho_rgb_shape[0]])
    h = np.min([10000, ortho_rgb_shape[1]])
    
    if ortho_rgb_shape[0] > 10000 or ortho_rgb_shape[1] > 10000:
        ortho_rgb = cv2.resize(ortho_rgb, (10000, 10000), interpolation=cv2.INTER_AREA)
    
    # Create output directories if they don't exist
    if not os.path.exists(config['rgb_older_vignettes_path'] + f'rgb_{year}_large/'):
        os.makedirs(config['rgb_older_vignettes_path'] + f'rgb_{year}_large/')
    if not os.path.exists(config['rgb_older_vignettes_path'] + f'rgb_{year}_large/rgb_{year}_{rgb_x}_{rgb_y}/'):
        os.makedirs(config['rgb_older_vignettes_path'] + f'rgb_{year}_large/rgb_{year}_{rgb_x}_{rgb_y}/')


    if create_stats:
        # Read existing DataFrame or create a new one
        if os.path.exists(config['stat_older_vignettes_path']):
            df = pd.read_csv(config['stat_older_vignettes_path'], index_col="Unnamed: 0")
        else:
            df = pd.DataFrame(columns=config['stat_older_vignettes_col'])

    if os.path.exists(config['ortho_shapes_path'] + 'large_ortho_shapes.json'):
        with open(config['ortho_shapes_path'] + 'large_ortho_shapes.json') as f:
            ortho_shapes = json.load(f)
    else:
        ortho_shapes = {}

    if os.path.exists(config['ortho_positions_path'] + 'large_ortho_positions.json'):
        with open(config['ortho_positions_path'] + 'large_ortho_positions.json') as f:
            ortho_positions = json.load(f)
    else:
        ortho_positions = {}
    
    shapes = {}
    positions = {}
    
    x_crops = [i*config['large_img_size'] for i in range(int(np.ceil(w/config['large_img_size'])))]
    y_crops = [i*config['large_img_size'] for i in range(int(np.ceil(h/config['large_img_size'])))]

    # range(0, w-(config['large_img_size'])+1, config['large_img_size']):
    for x in x_crops:
        # range(0, h-(config['large_img_size'])+1, config['large_img_size']):
        for y in y_crops:

            # print(x, y, np.max([0, y-config['border']]), y+config['large_img_size']+config['border'], np.max([0, x-config['border']]), x+config['large_img_size']+config['border'])
            crop_rgb = ortho_rgb[np.max([0, y]): y+config['large_img_size'], np.max([0, x]): x+config['large_img_size'], :]
            # ortho_rgb[np.max([0, y-config['border']]): y+config['large_img_size']+config['border'], np.max([0, x-config['border']]): x+config['large_img_size']+config['border'], :]
            # ortho_rgb[np.max([0, mnhc_y*h+y-config['border']]): mnhc_y*h+y+config['img_size']+config['border'], np.max([0, mnhc_x*w+x-config['border']]): mnhc_x*w+x+config['img_size']+config['border'], :]

            pos = dept+'_'+str(int(int(rgb_x)*1000+x/2))+'_'+str(int(int(rgb_y)*1000+y/2)) 
            #  dept+'_'+str(int((int(rgb_x)+mnhc_x)*1000+x/2))+'_'+str(int((int(rgb_y)-mnhc_y)*1000+y/2))

            cv2.imwrite(config['rgb_older_vignettes_path'] + f'rgb_{year}_large/rgb_{year}_{rgb_x}_{rgb_y}/' + f'rgb_{year}_large_' + str(pos) + '.jpg', crop_rgb)

            positions[pos] = tuple(np.array([np.max([0, y]), y+config['large_img_size'], np.max([0, x]), x+config['large_img_size']])/100)
            # tuple(np.array([np.max([0, y-config['border']]), y+config['large_img_size']+config['border'], np.max([0, x-config['border']]), x+config['large_img_size']+config['border']])/100)
            shapes[pos] = crop_rgb.shape
            
            if create_stats:
            # Calculate characteristics of the RGB crop and add to the DataFrame
                res = [year, pos, crop_rgb.sum()] + \
                    np.mean(crop_rgb, axis=(0, 1)).tolist() + \
                    np.std(crop_rgb, axis=(0, 1)).tolist() + \
                    [np.mean(convolution(crop_rgb[:, :, 0], N=3)), np.mean(convolution(crop_rgb[:, :, 1], N=3)), np.mean(convolution(crop_rgb[:, :, 2], N=3))] + \
                    [np.std(convolution(crop_rgb[:, :, 0], N=3)), np.std(convolution(crop_rgb[:, :, 1], N=3)), np.std(convolution(crop_rgb[:, :, 2], N=3))] 

                # Append the list to the DataFrame
                df.loc[len(df) + len(set(range(df.index[-1]))-set(df.index))
                if len(df.index) > 0
                else 0] = res



    ortho_shapes[rgb_name] = shapes
    with open('vignettes/rgb_older/large_ortho_shapes.json', 'w') as f:
        json.dump(ortho_shapes, f)

    ortho_positions[rgb_name] = positions
    with open('vignettes/rgb_older/large_ortho_positions.json', 'w') as f:
        json.dump(ortho_positions, f)
    
    if create_stats:
        # Drop duplicates and save the DataFrame to a CSV file
        df.drop_duplicates(subset='file').to_csv(config['stat_older_vignettes_path'])

        return ortho_positions, ortho_shapes, df
    
    else:
        return ortho_positions, ortho_shapes