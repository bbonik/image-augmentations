#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test the affine augmentation function, in an image with multiple
bounding boxes.

@author: vasileios vonikakis
"""


from augmentations import augment_affine
import matplotlib.pyplot as plt
plt.close('all')



if __name__=="__main__":
    
    
    filename = "../images/dogs_1.jpg"
    
    
    # reference test bboxes
    x1 = [320,805]
    y1 = [404,1078]
    w1 = x1[1] - x1[0]
    h1 = y1[1] - y1[0]
    
    x2 = [888,1296]
    y2 = [270,1078]
    w2 = x2[1] - x2[0]
    h2 = y2[1] - y2[0]
    
    x3 = [1300,1683]
    y3 = [450,1094]
    w3 = x3[1] - x3[0]
    h3 = y3[1] - y3[0]
    
    
    # getting augmented images
    image_augm = augment_affine(
        image_filename=filename,
        bboxes = [
            (x1[0], y1[0], w1, h1), 
            (x2[0], y2[0], w2, h2), 
            (x3[0], y3[0], w3, h3) 
            ],
        how_many=10,
        random_seed=0,
        range_scale=(0.4, 1.3),  # percentage
        range_translation=(-30, 30),  # in pixels
        range_rotation=(-30, 30),  # in degrees
        range_sheer=(-30, 30),  # in degrees
        enhance=False,
        bbox_truncate = True,
        bbox_discard_thr = 0.85,  # percentage
        verbose=True  # set as False if you are generating lots of images!!!
        )

    
    
    
    