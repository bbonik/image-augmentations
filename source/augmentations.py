#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for generating geometric affine augmentations in images, while also
handling bounding boxes.

@author: vasileios vonikakis
"""

import imageio
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import warp, AffineTransform
from skimage.exposure import equalize_adapthist




def flip_image(image, direction='lr'):
    # direction='lr' -> flip left right
    # direction='ud' -> flip up down

    image_flipped = image.copy()
    
    if direction == 'ud':
        image_flipped = np.flipud(image_flipped)
    else:
        image_flipped = np.fliplr(image_flipped)

    return image_flipped



def flip_bboxes(ls_bboxes, image_width, image_height, direction='lr'):
    # direction='lr' -> flip left right
    # direction='ud' -> flip up down

    ls_bboxes_flipped = []
    
    for bbox in ls_bboxes:
        
        if direction == 'ud':
            ls_bboxes_flipped.append(
                (
                    bbox[0],
                    image_height - (bbox[1] + bbox[3]),  # height-y_down_right
                    bbox[2], 
                    bbox[3]
                    )
                )
        else:
            ls_bboxes_flipped.append(
                (
                    image_width - (bbox[0] + bbox[2]),  # width-x_down_right
                    bbox[1], 
                    bbox[2], 
                    bbox[3]
                    )
                )
    
    return ls_bboxes_flipped






def augment_affine(
        image_filename,
        bboxes = None,  # list of tuples
        how_many=1,
        random_seed=0,
        range_scale=(0.8, 1.2),  # percentage
        range_translation=(-100, 100),  # in pixels
        range_rotation=(-45, 45),  # in degrees
        range_sheer=(-45, 45),  # in degrees
        flip_lr = None,
        flip_ud = None,
        enhance = False,
        bbox_truncate = True,
        bbox_discard_thr = 0.75,
        verbose=False
        ):
    
    '''
    ---------------------------------------------------------------------------
      Functiont that generates random affine augmentations for a given image
    ---------------------------------------------------------------------------
    The function apply affine distortions on a given image and generates many
    random variations of it. If bounding boxes are provided, then they are also
    transformed to the new distorted image and returned back.
    
    
    INPUTS
    ------
    image_filename: string
        Filename of the input image.
    bboxes: list of tuples or None
        The list of the given bounding boxes in the image. A bounding box is 
        defined by 4 numbers: upper left x (x), upper left y (y), width (w), 
        height (h). So if there are 2 bounding boxes in the image, bboxes 
        should be: [(x1,y1,w1,h1), (x2,y2,w2,h2)]
    how_many: int
        How many augmentations to generate per input image.
    random_seed: int
        Number for setting the random number generator, in order to
    range_scale: tuple of float
        Minimum and maximum range of possible scale factors (min, max)
    range_translation: tuple of int
        Minimum and maximum range of possible translation shifts in pixels
    range_rotation: tuple of int
        Minimum and maximum range of possible rotations in degrees
    range_sheer: tuple of int
        Minimum and maximum range of possible rotations in degrees
    flip_lr: string or None
        None: no left-right flipping is applied.
        'all': all images are flipped left-to-right (doubles number of images)
        'random': images are flipped left-to-right randomly
    flip_ud: string or None
        None: no up-down flipping is applied.
        'all': all images are flipped up-to-down (doubles number of images)
        'random': images are flipped up-to-down randomly
    enhance: boolean
        Use or not CLAHE enhancement (Contrast Limited Adaptive Histogram 
        Equalization)
    bbox_truncate: bool
        Whether or not to truncate bounding boxes within image boundaries.
    bbox_discard_thr: float [0,1]
        Helps to discard any new bounding boxes that are located mostly 
        outside the image boundaries, due to the augmentations. If the ratio 
        of the surface of a new bounding box (after image augmentation), over 
        the surface of the original bounding box, is less than 
        bbox_discard_thr, then the new bounding box is discarded (i.e. object 
        lies mostly outside the new image). Values closer to 1 are more strict
        whereas values closer to 0 are more permissive. 
    verbose: boolean
        Show visualizations and details or not.
    
    OUTPUT
    ------
    dcionary containing:
        Augmented images.
        Transformed bounding boxes.
        Details about each augmentation. 
        
    '''
    
    
    # load image
    image = imageio.imread(image_filename)
    
    if enhance is True:
        image = equalize_adapthist(
            image, 
            kernel_size=None, 
            clip_limit=0.01, 
            nbins=256
        )

    # show original image
    if verbose is True:
        print('\nAugmenting', image_filename.split('/')[-1], end='')
        print(' [x', end='')
        print(how_many, end='')
        if flip_lr=='all': print(' x2', end='')
        if flip_ud=='all': print(' x2', end='')
        print(']')
        plt.figure()
        plt.imshow(image, vmin=0, vmax=255)
        plt.title('Original')
        plt.axis('off')
        plt.tight_layout(True)
        if bboxes is not None:
            for bbox in bboxes:
                plt.gca().add_patch(
                    Rectangle(
                        xy=(bbox[0],bbox[1]),
                        width=bbox[2],
                        height=bbox[3],
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none'
                        )
                    )
        plt.show()
        
        

    # convert bboxes to x,y coordinates
    if bboxes is not None:
        
        ls_bboxes_coord = []
        
        for i in range(len(bboxes)):
            
            x_up_left = bboxes[i][0]
            y_up_left = bboxes[i][1]
            x_down_right = x_up_left + bboxes[i][2]  # x_up_left + width
            y_down_right = y_up_left + bboxes[i][3]  # y_up_left + height
            
            ls_bboxes_coord.append([
                [x_up_left, y_up_left], 
                [x_down_right, y_up_left], 
                [x_down_right, y_down_right], 
                [x_up_left, y_down_right]
                ])
     
    #------------------------------------------------------- get random values
        
    # set random seed
    np.random.seed(random_seed)  
    
    # degrees to radians
    range_rotation = np.radians(range_rotation)
    range_sheer = np.radians(range_sheer)
    
    # get random values
    param_scale = np.random.uniform(
        low=range_scale[0], 
        high=range_scale[1], 
        size=how_many
        )
    param_trans = np.random.uniform(
        low=range_translation[0], 
        high=range_translation[1], 
        size=(how_many,2)
        ).astype(int)
    param_rot = np.random.uniform(
        low=range_rotation[0], 
        high=range_rotation[1], 
        size=how_many
        )
    param_sheer = np.random.uniform(
        low=range_sheer[0], 
        high=range_sheer[1], 
        size=how_many
        )
    
    #-------------------------------------------- process all image variations
    
    # initiate output dcionary
    dc_augm = {}
    dc_augm['Images'] = []
    dc_augm['bboxes'] = []
    dc_augm['Transformations'] = []
    
    
    # for all images
    for i in range(how_many):
        
        dc_augm['bboxes'].append([])
    
        # configure an affine transform based on the random values
        tform = AffineTransform(
                scale=(param_scale[i],param_scale[i]),          
                rotation=param_rot[i], 
                shear=param_sheer[i],
                translation=(param_trans[i,0], param_trans[i,1])
                )
        
        image_transformed = warp(   # warp image (pixel range -> float [0,1])
                image,       
                tform.inverse, 
                mode = 'symmetric'
                )
        
        # convert range back to [0,255]
        image_transformed *= 255
        image_transformed = image_transformed.astype(np.uint8)
        
        # add transforamtions to the dictionary 
        dc_augm['Images'].append(image_transformed)
        
        dc_transf = {}
        dc_transf['Scale'] = param_scale[i]
        dc_transf['Translation'] = param_trans[i]
        dc_transf['Rotation'] = np.degrees(param_rot[i])
        dc_transf['Sheer'] = np.degrees(param_sheer[i])
        dc_transf['Flip_lr'] = False
        dc_transf['Flip_ud'] = False
        dc_transf['Matrix'] = tform.params
        
        dc_augm['Transformations'].append(dc_transf)
          
    #------------- transform bboxes to the new coordinates of the warped image
    
        if bboxes is not None:
            
            ls_bboxes_coord_new = copy.deepcopy(ls_bboxes_coord)
            
            for b in range(len(ls_bboxes_coord_new)):
                
                for j in range(4):
                    ls_bboxes_coord_new[b][j].append(1)  # [x,y,1]
                    vector = np.array(ls_bboxes_coord_new[b][j])  
                    new_coord = np.matmul(tform.params, vector)
                    ls_bboxes_coord_new[b][j][0] = int(round(new_coord[0]))# x
                    ls_bboxes_coord_new[b][j][1] = int(round(new_coord[1]))# y

                # get the final bboxes from the new (transformed) coordinates
                # (find the min and max of the transformed xy coordinates)
                # TODO: add a diminishing factor for skewed bboxes to address
                # the fact that bboxes from highly skewed images expand!
                x_up_left = min(
                    ls_bboxes_coord_new[b][0][0], 
                    ls_bboxes_coord_new[b][1][0], 
                    ls_bboxes_coord_new[b][2][0], 
                    ls_bboxes_coord_new[b][3][0]
                    )
                y_up_left = min(
                    ls_bboxes_coord_new[b][0][1], 
                    ls_bboxes_coord_new[b][1][1], 
                    ls_bboxes_coord_new[b][2][1], 
                    ls_bboxes_coord_new[b][3][1]
                    )
                x_down_right = max(
                    ls_bboxes_coord_new[b][0][0], 
                    ls_bboxes_coord_new[b][1][0], 
                    ls_bboxes_coord_new[b][2][0], 
                    ls_bboxes_coord_new[b][3][0]
                    )
                y_down_right = max(
                    ls_bboxes_coord_new[b][0][1], 
                    ls_bboxes_coord_new[b][1][1], 
                    ls_bboxes_coord_new[b][2][1], 
                    ls_bboxes_coord_new[b][3][1]
                    )
                
    #--------------------------------------- truncate bbox to image boundaries
                
                flag_truncated = False
                
                if bbox_truncate is True:
                    
                    im_width = image_transformed.shape[1]
                    im_height = image_transformed.shape[0]
                    
                    if x_up_left < 0: 
                        x_up_left = 0
                        flag_truncated = True
                    elif x_up_left > im_width: 
                        x_up_left = im_width - 1
                        flag_truncated = True
                    
                    if x_down_right < 0: 
                        x_down_right = 0
                        flag_truncated = True
                    elif x_down_right > im_width: 
                        x_down_right = im_width - 1
                        flag_truncated = True
                    
                    if y_up_left < 0: 
                        y_up_left = 0
                        flag_truncated = True
                    elif y_up_left > im_height: 
                        y_up_left = im_height - 1
                        flag_truncated = True
                    
                    if y_down_right < 0: 
                        y_down_right = 0
                        flag_truncated = True
                    elif y_down_right > im_height: 
                        y_down_right = im_height - 1
                        flag_truncated = True

                
                width_new = x_down_right - x_up_left
                height_new = y_down_right - y_up_left
                width_old = bboxes[b][2]
                height_old = bboxes[b][3]
                
                # estimate how much the bbox area has changed due to cropping
                bbox_surface_ratio = ((width_new * height_new) / 
                                     (width_old * height_old * param_scale[i]))

    #------------------------------------------------------ store the new bbox
                
                if ((flag_truncated == False) | 
                    (bbox_surface_ratio > bbox_discard_thr)):
                    
                    dc_augm['bboxes'][i].append(
                        (
                            x_up_left,
                            y_up_left, 
                            width_new, 
                            height_new
                            )
                        )

    #-------------------------------------------------- flip images left-right
    
    if flip_lr is not None:
        
        if flip_lr == 'random':
            
            for i in range(len(dc_augm['Images'])):
            
                if random.choice([True, False]):  # random boolean
                    
                    dc_augm['Transformations'][i]['Flip_lr'] = True
            
                    dc_augm['Images'][i] = flip_image(
                        image=dc_augm['Images'][i],
                        direction='lr'
                        )
                    
                    if bboxes is not None: 
                        dc_augm['bboxes'][i] = flip_bboxes(
                            ls_bboxes=dc_augm['bboxes'][i],
                            image_width=dc_augm['Images'][i].shape[1],
                            image_height=dc_augm['Images'][i].shape[0],
                            direction='lr'
                            )
                    
        else:  # flip_lr == 'all':
            
            ls_flipped_images = []
            ls_flipped_bboxes = []
            dc_augm['Transformations'].extend(
                copy.deepcopy(
                    dc_augm['Transformations']
                    )
                )
            
            for i in range(len(dc_augm['Images'])):
                    
                dc_augm['Transformations'][i + how_many]['Flip_lr'] = True
        
                ls_flipped_images.append(
                    flip_image(
                        image=dc_augm['Images'][i],
                        direction='lr'
                        )
                    )
                
                ls_flipped_bboxes.append(
                    flip_bboxes(
                        ls_bboxes=dc_augm['bboxes'][i],
                        image_width=dc_augm['Images'][i].shape[1],
                        image_height=dc_augm['Images'][i].shape[0],
                        direction='lr'
                        )
                    )
            
            dc_augm['Images'].extend(ls_flipped_images)
            dc_augm['bboxes'].extend(ls_flipped_bboxes)

    #----------------------------------------------------- flip images up-down
    
    if flip_ud is not None:
        
        if flip_ud == 'random':
            
            for i in range(len(dc_augm['Images'])):
            
                if random.choice([True, False]):  # random boolean
                    
                    dc_augm['Transformations'][i]['Flip_ud'] = True
            
                    dc_augm['Images'][i] = flip_image(
                        image=dc_augm['Images'][i],
                        direction='ud'
                        )
                    
                    if bboxes is not None: 
                        dc_augm['bboxes'][i] = flip_bboxes(
                            ls_bboxes=dc_augm['bboxes'][i],
                            image_width=dc_augm['Images'][i].shape[1],
                            image_height=dc_augm['Images'][i].shape[0],
                            direction='ud'
                            )
                    
        else:  # flip_ud == 'all':
            
            ls_flipped_images = []
            ls_flipped_bboxes = []
            dc_augm['Transformations'].extend(
                copy.deepcopy(
                    dc_augm['Transformations']
                    )
                )
            
            for i in range(len(dc_augm['Images'])):
                    
                dc_augm['Transformations'][i + how_many]['Flip_ud'] = True
        
                ls_flipped_images.append(
                    flip_image(
                        image=dc_augm['Images'][i],
                        direction='ud'
                        )
                    )
                
                ls_flipped_bboxes.append(
                    flip_bboxes(
                        ls_bboxes=dc_augm['bboxes'][i],
                        image_width=dc_augm['Images'][i].shape[1],
                        image_height=dc_augm['Images'][i].shape[0],
                        direction='ud'
                        )
                    )
            
            dc_augm['Images'].extend(ls_flipped_images)
            dc_augm['bboxes'].extend(ls_flipped_bboxes)
                    
    #------------------------------------------------- visualize augmentations

    if verbose is True:
        
        for i,image_transformed in enumerate(dc_augm['Images']):
        
            print('\nTransformation for augmentation', i+1)
            print('Scale:', 
                  dc_augm['Transformations'][i]['Scale'])
            print('Translation_x:', 
                  dc_augm['Transformations'][i]['Translation'][0])
            print('Translation_y:', 
                  dc_augm['Transformations'][i]['Translation'][1])
            print('Rotation:', 
                  dc_augm['Transformations'][i]['Rotation'])
            print('Sheer:', 
                  dc_augm['Transformations'][i]['Sheer'])
            print('Flip left->right: ', 
                  dc_augm['Transformations'][i]['Flip_lr'])
            print('Flip up->down: ', 
                  dc_augm['Transformations'][i]['Flip_ud'])
            
            plt.figure()
            plt.imshow(image_transformed, interpolation='bilinear')
            plt.title('Augmented #' + str(i+1))
            plt.tight_layout(True)
            plt.axis('off')
            
            if bboxes is not None:
                for bbox in dc_augm['bboxes'][i]:
                    plt.gca().add_patch(
                    Rectangle(
                        xy=(bbox[0],bbox[1]),
                        width=bbox[2],
                        height=bbox[3],
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none'
                        )
                    )
            plt.show()
    
    
    if bboxes is None: del dc_augm['bboxes']
    
    return dc_augm

