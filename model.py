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


def build_decoder_block(pl, n_filters, k_size_tr, k_size , strides, padding, af):
    """
    Decoder path convolution block builder
    :param pl: previous layer
    :param n_filters: number of filters in Conv3D and Conv3DTranspose layer
    :param k_size_tr: kernel size in Conv3DTranspose layer
    :param k_size: kernel size in Conv3D layer
    :param strides: strides in Conv3DTranspose layer
    :param padding: same
    :param af: activation function in Conv3D layer
    :return: Upsampling block
    """
    tr_1 = Conv3DTranspose(filters=n_filters, kernel_size=k_size_tr, strides=strides, padding=padding)(pl)
    conv_1 = Conv3D(filters=n_filters, kernel_size=k_size, padding=padding, activation=af)(tr_1)
    concat_1 = concatenate([tr_1, conv_1], axis=4)
    conv_2 = Conv3D(filters=n_filters, kernel_size=k_size, padding=padding, activation=af)(concat_1)
    concat_2 = concatenate([tr_1, conv_2], axis=4)
    return concat_2


def build_bridge_block(pl, n_filters=512, k_size=(3, 3, 3), padding="same", af=relu):
    """
    Bridge convolution block builder
    :param pl: previous layer
    :param n_filters: number of filters in convolution layer, default is 512
    :param k_size: kernel size in convolution layer, default is (3, 3, 3)
    :param padding: default is same
    :param af: activation function in convolution layer, default is relu
    :return: convolution block
    """
    conv_1 = Conv3D(filters=n_filters, kernel_size=k_size, padding=padding, activation=af)(pl)
    concat_1 = concatenate([pl, conv_1], axis=4)
    conv_2 = Conv3D(filters=n_filters, kernel_size=k_size, padding=padding, activation=af)(concat_1)
    concat_2 = concatenate([pl, conv_2], axis=4)
    return concat_2


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
bb = build_bridge_block(pl=cb_4)


# Defining decoder path layers
# First block
db_1 = build_decoder_block(pl=bb, n_filters=256, k_size_tr=(2, 2, 2),
                           k_size=(3, 3, 3), strides=BRAIN, padding="same", af=relu)
# First skip connection
sc_1 = concatenate([cb_3, db_1], axis=4)
# Second block
db_2 = build_decoder_block(pl=sc_1, n_filters=128, k_size_tr=(2, 2, 2),
                           k_size=(3, 3, 3), strides=BRAIN, padding="same", af=relu)
# Second skip connection
sc_2 = concatenate([cb_2, db_2], axis=4)
# Third block
db_3 = build_decoder_block(pl=sc_2, n_filters=64, k_size_tr=(2, 2, 2),
                           k_size=(3, 3, 3), strides=BRAIN, padding="same", af=relu)
# Third skip connection
sc_3 = concatenate([cb_1, db_3], axis=4)
# Forth block
db_4 = build_decoder_block(pl=sc_3, n_filters=32, k_size_tr=(2, 2, 2),
                           k_size=(3, 3, 3), strides=BRAIN, padding="same", af=relu)

# Output layer
out_layer = Conv3D(filters=1, kernel_size=(1, 1, 1), activation=sigmoid)(db_4)

# Instantiating the model
model = Model(inputs=[in_layer], outputs=[out_layer], name="3D Dense U-Net")
model.summary()
