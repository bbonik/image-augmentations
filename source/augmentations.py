#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for generating geometric affine augmentations in images, while also
handling bounding boxes.

@author: vasileios vonikakis
"""

import imageio
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import warp, AffineTransform
from skimage.exposure import equalize_adapthist






def augment_affine(
        image_filename,
        bboxes = None,  # list of tuples
        how_many=1,
        random_seed=0,
        range_scale=(0.8, 1.2),  # percentage
        range_translation=(-100, 100),  # in pixels
        range_rotation=(-45, 45),  # in degrees
        range_sheer=(-45, 45),  # in degrees
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
    enhance: boolean
        Use or not CLAHE enhancement (Contrast Limited Adaptive Histogram 
        Equalization)
    bbox_truncate: bool
        Whether or not to truncate bounding boxes within image boundaries.
    bbox_discard_thr: float [0,1]
        If the ratio of the surface of a new bounding box (after image 
        augmentation), over the surface of the original bounding box, is less
        than bbox_discard_thr, then the new bounding box is discarded. This 
        parameter helps to filter out any bounding boxes that lie mostly 
        outsite the image boundaries. 
    verbose: boolean
        Show visualizations and details or not.
    
    OUTPUT
    ------
    Dictionary containing:
        Augmented images.
        Transformed bounding boxes.
        Transformation matrices used for each augmentation. 
        
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
        print('\nAugmenting', image_filename.split('/')[-1])
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
    
    
    # initiate output dictionary
    dict_augmentations = {}
    dict_augmentations['Images'] = []
    dict_augmentations['Matrices'] = []
    dict_augmentations['bboxes'] = []
    
    
    # for all images
    for i in range(how_many):
        
        dict_augmentations['bboxes'].append([])
    
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
        
        # add to the list
        dict_augmentations['Images'].append(image_transformed)
        dict_augmentations['Matrices'].append(tform.params)
        
        
        # transform bboxes to the new coordinates of the warped image
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
                
                if bbox_truncate is True:
                    # truncating bboxes to the image boundaries
                    
                    im_width = image_transformed.shape[1]
                    im_height = image_transformed.shape[0]
                    
                    flag_truncated = False
                    
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
                    
                    # estimatge how much the bbox surface has changed
                    bbox_surface_ratio = ((width_new * height_new) / 
                                          (width_old * height_old))

                    if (flag_truncated == False) | (bbox_surface_ratio > bbox_discard_thr):
                            dict_augmentations['bboxes'][i].append(
                                (
                                    x_up_left,
                                    y_up_left, 
                                    width_new, 
                                    height_new
                                    )
                                )

                else:
                    # return new bboxes, even if they lie outside the image
                    width_new = x_down_right - x_up_left
                    height_new = y_down_right - y_up_left
                    dict_augmentations['bboxes'][i].append(
                        (
                            x_up_left,
                            y_up_left, 
                            width_new, 
                            height_new
                            )
                        )


        if verbose is True:
            print('\nTransformation for augmentation', i)
            print('Scale:',param_scale[i])
            print('Translation_x:',param_trans[i,0], 
                  'Translation_y:',param_trans[i,1])
            print('Rotation:',param_rot[i])
            print('Sheer:',param_sheer[i])
            
            plt.figure()
            plt.imshow(image_transformed, interpolation='bilinear')
            plt.title('Augmented #' + str(i+1))
            plt.tight_layout(True)
            plt.axis('off')
            
            if bboxes is not None:
                for bbox in dict_augmentations['bboxes'][i]:
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
    
    
    
    return dict_augmentations

