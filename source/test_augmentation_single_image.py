#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test the affine augmentation function, in an image with multiple
bounding boxes.

@author: vasileios vonikakis
"""

import json
from augmentations import augment_affine
import matplotlib.pyplot as plt
plt.close('all')



if __name__=="__main__":
    
    
    # open json file with bounding boxes (x,y,w,h)
    with open('../images/bboxes.json', 'r') as json_file:
       dc_bboxes = json.load(json_file)
       
    
    # get filename and list of bounding boxes
    filename = dc_bboxes['image1']['filename']
    ls_bboxes = dc_bboxes['image1']['bboxes']
    
    # filename = dc_bboxes['image2']['filename']
    # ls_bboxes = dc_bboxes['image2']['bboxes']
    
    
    # getting augmented images
    image_augm = augment_affine(
        image_filename=filename,
        # bboxes = None,
        bboxes =ls_bboxes,
        how_many=10,
        random_seed=0,
        range_scale=(0.5, 1.5),  # percentage
        range_translation=(-30, 30),  # in pixels
        range_rotation=(-30, 30),  # in degrees
        range_sheer=(-30, 30),  # in degrees
        flip_lr='random',
        flip_ud=None,
        enhance=False,
        bbox_truncate = True,
        bbox_discard_thr = 0.85,  # percentage
        verbose=True  # set as False if you are generating lots of images!!!
        )
    
    