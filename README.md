# image-augmentations
Simple function for generating affine geometric augmentations on images and bounding boxes. It can be used to augment image datasets and generate multiple variations from a single image.

![overview](images/overview.jpg "overview")

- **augment_affine()**: Generates multiple affine variations of an input image, while adjusting any existing bounding boxes.

# Contents:
```tree
├── source                         [Directory: Source code]
│   ├── augmentations.py           [Main script with all the functions]  
│   └── test_augmentation.py       [Example for testing augmentations]
└── images                         [Directory: Sample test images]
```

# Dependences
- numpy
- skimage
- matplotlib
