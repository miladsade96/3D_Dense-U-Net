"""
    3D Dense U-Net Model Implementation
    Paper url: https://www.mdpi.com/2076-3417/9/3/404
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""


import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import (Conv3DTranspose, Conv3D, MaxPooling3D, concatenate, Input)


# Setting image data format
K.set_image_data_format("channels_last")

# Kernel(Pool) Size
BRAIN = (2, 2, 2)
SPINE = (1, 2, 2)

# Image Dimensions
img_height = 256
img_width = 256
img_depth = 256
channels = 3


# Defining input layer
in_layer = Input((img_height, img_width, img_depth, channels))


# Defining encoder path layers
# First block
conv_11 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding="same", activation=relu)(in_layer)
# Concatenation of input layer and conv_11
concat_1 = concatenate([in_layer, conv_11], axis=4)
conv_12 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding="same", activation=relu)(concat_1)
# Concatenation of input layer and conv_12
concat_2 = concatenate([in_layer, conv_12], axis=4)
mp_1 = MaxPooling3D(pool_size=BRAIN, strides=(2, 2, 2), padding="same")(concat_2)

# Second block
conv_21 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=relu)(mp_1)
# Concatenation of mp_1, conv_21
concat_3 = concatenate([mp_1, conv_21], axis=4)
conv_22 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=relu)(concat_3)
# Concatenation of mp_1, conv_22
concat_4 = concatenate([mp_1, conv_22], axis=4)
mp_2 = MaxPooling3D(pool_size=BRAIN, strides=(2, 2, 2), padding="same")(concat_4)

# Third block
conv_31 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding="same", activation=relu)(mp_2)
# Concatenation of mp_2, conv_31
concat_5 = concatenate([mp_2, conv_31], axis=4)
conv_32 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding="same", activation=relu)(concat_5)
# Concatenation of mp_2, conv_32
concat_6 = concatenate([mp_2, conv_32], axis=4)
mp_3 = MaxPooling3D(pool_size=BRAIN, strides=(2, 2, 2), padding="same")(concat_6)

# Fourth block
conv_41 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding="same", activation=relu)(mp_3)
# Concatenation of mp_3, conv_41
concat_7 = concatenate([mp_3, conv_41], axis=4)
conv_42 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding="same", activation=relu)(concat_7)
# Concatenation of mp_3, conv_42
concat_8 = concatenate([mp_3, conv_42], axis=4)
mp_4 = MaxPooling3D(pool_size=BRAIN, strides=(2, 2, 2), padding="same")(concat_8)


# Defining the bridge block
conv_51 = Conv3D(filters=512, kernel_size=(3, 3, 3), padding="same", activation=relu)(mp_4)
# Concatenation of mp_4, conv_51
concat_9 = concatenate([mp_4, conv_51], axis=4)
conv_52 = Conv3D(filters=512, kernel_size=(3, 3, 3), padding="same", activation=relu)(concat_9)
# Concatenation of mp_4, conv_52
concat_10 = concatenate([mp_4, conv_52], axis=4)


# Defining decoder path layers
# First block
tr_1 = Conv3DTranspose(filters=256, kernel_size=(2, 2, 2), strides=BRAIN, padding="same")(concat_10)
conv_61 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding="same", activation=relu)(tr_1)
# Concatenation of tr_1, conv_61
concat_11 = concatenate([tr_1, conv_61], axis=4)
conv_62 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding="same", activation=relu)(concat_11)
# Concatenation of tr_1, conv_62
concat_12 = concatenate([tr_1, conv_62], axis=4)

# First skip connection
sc_1 = concatenate([mp_3, concat_12], axis=4)

# Second block
tr_2 = Conv3DTranspose(filters=128, kernel_size=(2, 2, 2), strides=BRAIN, padding="same")(sc_1)
conv_71 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding="same", activation=relu)(tr_2)
# Concatenation of tr_2, conv_71
concat_13 = concatenate([tr_2, conv_71], axis=4)
conv_72 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding="same", activation=relu)(concat_13)
# Concatenation of tr_2, conv_72
concat_14 = concatenate([tr_2, conv_72], axis=4)

# Second skip connection
sc_2 = concatenate([mp_2, concat_14], axis=4)

# Third block
tr_3 = Conv3DTranspose(filters=64, kernel_size=(2, 2, 2), strides=BRAIN, padding="same")(sc_2)
conv_81 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=relu)(tr_3)
# Concatenation of tr_3, conv_81
concat_15 = concatenate([tr_3, conv_81], axis=4)
conv_82 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", activation=relu)(concat_15)
# Concatenation of tr_3, conv_82
concat_16 = concatenate([tr_3, conv_82], axis=4)

# Third skip connection
sc_3 = concatenate([mp_1, concat_16], axis=4)

# Forth block
tr_4 = Conv3DTranspose(filters=32, kernel_size=(2, 2, 2), strides=BRAIN, padding="same")(sc_3)
conv_91 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding="same", activation=relu)(tr_4)
# Concatenation of tr_4, conv_91
concat_17 = concatenate([tr_4, conv_91], axis=4)
conv_92 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding="same", activation=relu)(concat_17)
# Concatenation of tr_4, conv_92
concat_18 = concatenate([tr_4, conv_92], axis=4)

# Output layer
out_layer = Conv3D(filters=1, kernel_size=(1, 1, 1), activation=sigmoid)(concat_18)

# Instantiating the model
model = Model(inputs=[in_layer], outputs=[out_layer], name="3D Dense U-Net")
model.summary()
