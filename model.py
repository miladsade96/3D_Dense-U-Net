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
