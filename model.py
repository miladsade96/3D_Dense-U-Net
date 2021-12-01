"""
    3D Dense U-Net Model Implementation
    Paper url: https://www.mdpi.com/2076-3417/9/3/404
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""


import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv3DTranspose, Conv3D, MaxPooling3D, concatenate, Input)


# Setting image data format
K.set_image_data_format("channels_last")

# Kernel(Pool) Size
BRAIN = (2, 2, 2)
SPINE = (1, 2, 3)

# Image Dimensions
img_height = 256
img_width = 256
img_depth = 256
channels = 3


# Defining input layer
in_layer = Input((img_height, img_width, img_depth, channels))
