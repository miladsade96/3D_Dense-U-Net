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
img_height = 32
img_width = 32
img_depth = 32
channels = 1


def build_encoder_block(pl, n_filters, k_size, padding, af, p_size, strides):
    """
    Encoder path convolution block builder
    :param pl: previous layer
    :param n_filters: number of filters in convolution layer
    :param k_size: kernel size in convolution layer
    :param padding: same
    :param af: activation function in convolution layer
    :param p_size: pool size in downsampling layer
    :param strides: strides in downsampling layer
    :return: convolution block
    """
    conv_1 = Conv3D(filters=n_filters, kernel_size=k_size, padding=padding, activation=af)(pl)
    concat_1 = concatenate([pl, conv_1], axis=4)
    conv_2 = Conv3D(filters=n_filters, kernel_size=k_size, padding=padding, activation=af)(concat_1)
    concat_2 = concatenate([pl, conv_2], axis=4)
    mp = MaxPooling3D(pool_size=p_size, strides=strides, padding=padding)(concat_2)
    return mp


# Defining input layer
in_layer = Input((img_height, img_width, img_depth, channels))


# Defining encoder path layers
# First block
cb_1 = build_encoder_block(pl=in_layer, n_filters=32, k_size=(3, 3, 3), padding="same",
                           af=relu, p_size=BRAIN, strides=(2, 2, 2))

# Second block
cb_2 = build_encoder_block(pl=cb_1, n_filters=64, k_size=(3, 3, 3), padding="same",
                           af=relu, p_size=BRAIN, strides=(2, 2, 2))

# Third block
cb_3 = build_encoder_block(pl=cb_2, n_filters=128, k_size=(3, 3, 3), padding="same",
                           af=relu, p_size=BRAIN, strides=(2, 2, 2))

# Fourth block
cb_4 = build_encoder_block(pl=cb_3, n_filters=256, k_size=(3, 3, 3), padding="same",
                           af=relu, p_size=BRAIN, strides=(2, 2, 2))


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
