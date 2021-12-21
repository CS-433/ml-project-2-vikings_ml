""" Implementation of our original UNet. """

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, BatchNormalization, MaxPool2D, Concatenate, Add


class UNet():
    """ U-net neural network architecture"""

    def __init__(self, filter_num=64, batch_norm=True, pad="same"):
        self.filter_num = filter_num
        self.batch_norm = batch_norm
        self.pad = pad

    def conv_3x3(self, input, depth):
        """ Convolving the input with a 3x3 kernel twice. """

        # First convolution, apply batch-normalization if chosen, set activation function to ReLU
        conv1 = Conv2D(2**depth * self.filter_num, (3, 3),
                       padding=self.pad, kernel_initializer='he_normal')(input)
        if self.batch_norm:
            conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)

        # Second convolution, same procedure using the output of the first convolution
        conv2 = Conv2D(self.filter_num, (3, 3), padding=self.pad)(conv1)
        if self.batch_norm:
            conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)

        return conv2

    def encoder(self, input, depth):
        """ For a given layer at a certain depth, performs first the convolution operation and then a 2x2 max pool. 
        Returns the 2x2 max pool as well as a copy of the convolution to be used by the decoderdepth=0 means top. """

        # Convolve and then maxpool
        saved_layer = self.conv_3x3(input, depth)
        conv_maxpool = MaxPool2D(strides=(2, 2))(saved_layer)

        return saved_layer, conv_maxpool

    def res(self, depth, kernelsize=(3, 3)):
        """ Creating a residual block. """
        model = keras.Sequential([
            Conv2D(2**depth * self.filter_num, kernelsize,
                   activation='relu', padding=self.pad),
            BatchNormalization(),
            Conv2D(2**depth * self.filter_num, kernelsize, padding=self.pad)])
        return model

    def encoder_ladder(self, input, depth, func, add=[], first=False):
        """ Encoder for the UNet. """

        # Add convolution, residual block, maxpool
        x = Conv2D(2**depth * self.filter_num, (3, 3),
                   padding=self.pad, kernel_initializer='he_normal')(input)
        print(f"depth: {depth}")

        # Add if not the first Unet
        if not first:
            x = Add()([x, add])

        # Shared weights
        fx = func(x)
        out = Add()([x, fx])
        out = ReLU()(out)
        saved_layer = BatchNormalization()(out)

        conv_maxpool = MaxPool2D(strides=(2, 2))(saved_layer)

        return saved_layer, conv_maxpool

    def decoder(self, input, saved_layer, depth):
        """ Decoder for the expansive path of the UNet (right). """

        # 2x2 convolution (“up-convolution”), upsampling
        up_conv = Conv2DTranspose(
            2**depth * self.filter_num, (2, 2), strides=2, padding="same")(input)

        # Concatenation with the correspondingly (cropped) feature map from the contracting path
        up_conv_cat = Concatenate()([up_conv, saved_layer])

        # 3x3 convolution
        result = self.conv_3x3(up_conv_cat, depth)

        return result

    def decoder_ladder(self, input, saved_layer, depth, func):
        """ Decoder for the expansive path of the ladderNet(right). """

        # 2x2 convolution (“up-convolution”), upsampling
        up_conv = Conv2DTranspose(
            2**depth * self.filter_num, (3, 3), strides=2, padding="same")(input)

        # Concatenation with the correspondingly (cropped) feature map from the contracting path
        x = Add()([up_conv, saved_layer])

        # Shared resblock
        fx = func(x)
        out = Add()([x, fx])
        out = ReLU()(out)
        result = BatchNormalization()(out)

        return result

    def get_model(self, image_shape):
        """ Creating the U-net architecture for a picture with a given input size. """

        input = keras.Input(shape=image_shape)

        # Contracting path (left side) of the network architecture
        saved_layer0, left0 = self.encoder(input, 0)
        saved_layer1, left1 = self.encoder(left0, 1)
        saved_layer2, left2 = self.encoder(left1, 2)
        saved_layer3, left3 = self.encoder(left2, 3)

        # Convolution between contracting and expansive path
        middle = self.conv_3x3(left3, 4)

        # Expansive path (right side) of the network architecture
        right3 = self.decoder(middle, saved_layer3, 3)
        right2 = self.decoder(right3, saved_layer2, 2)
        right1 = self.decoder(right2, saved_layer1, 1)
        right0 = self.decoder(right1, saved_layer0, 0)

        # Convolute to 2d picture
        output = Conv2D(1, (1, 1), padding="same",
                        activation="sigmoid")(right0)

        model = keras.models.Model(input, output)
        return model

    def get_model_ladder(self, image_shape):
        """ Creating the ladder-net architecture (consisting of two U-nets) for a picture with a given input size. """

        input = keras.Input(shape=image_shape)

        # Contracting path (left side) of the network architecture
        shared_resnets = [self.res(i) for i in range(5)]
        print(shared_resnets)
        saved_layer0, left0 = self.encoder_ladder(
            input, 0, shared_resnets[0], first=True)
        saved_layer1, left1 = self.encoder_ladder(
            left0, 1, shared_resnets[1], first=True)
        saved_layer2, left2 = self.encoder_ladder(
            left1, 2, shared_resnets[2], first=True)
        saved_layer3, left3 = self.encoder_ladder(
            left2, 3, shared_resnets[3], first=True)

        # Convolution between contracting and expansive path
        middle = Conv2D(2**4 * self.filter_num, (3, 3),
                        padding=self.pad, kernel_initializer='he_normal')(left3)
        fx = shared_resnets[4](middle)
        out = Add()([middle, fx])
        out = ReLU()(out)
        middle = BatchNormalization()(out)

        # Expansive path (right side) of the network architecture
        right3 = self.decoder_ladder(
            middle, saved_layer3, 3, shared_resnets[3])
        right2 = self.decoder_ladder(
            right3, saved_layer2, 2, shared_resnets[2])
        right1 = self.decoder_ladder(
            right2, saved_layer1, 1, shared_resnets[1])
        right0 = self.decoder_ladder(
            right1, saved_layer0, 0, shared_resnets[0])

        # First link
        fx = shared_resnets[0](right0)
        out = Add()([right0, fx])
        out = ReLU()(out)
        link = BatchNormalization()(out)

        # Second link
        fx = shared_resnets[0](link)
        out = Add()([link, fx])
        out = ReLU()(out)
        link1 = BatchNormalization()(out)

        # First row of second
        saved_layer0 = Add()([link1, right0])
        left0 = MaxPool2D(strides=(2, 2))(saved_layer0)

        # Contracting path (left side) of the second unet
        saved_layer1, left1 = self.encoder_ladder(
            left0, 1, shared_resnets[1], right1)
        saved_layer2, left2 = self.encoder_ladder(
            left1, 2, shared_resnets[2], right2)
        saved_layer3, left3 = self.encoder_ladder(
            left2, 3, shared_resnets[3], right3)

        # Middle
        middle = Conv2D(2**4 * self.filter_num, (3, 3),
                        padding=self.pad, kernel_initializer='he_normal')(left3)
        fx = shared_resnets[4](middle)
        out = Add()([middle, fx])
        out = ReLU()(out)
        middle = BatchNormalization()(out)

        # Expansive path
        right3 = self.decoder_ladder(
            middle, saved_layer3, 3, shared_resnets[3])
        right2 = self.decoder_ladder(
            right3, saved_layer2, 2, shared_resnets[2])
        right1 = self.decoder_ladder(
            right2, saved_layer1, 1, shared_resnets[1])
        right0 = self.decoder_ladder(
            right1, saved_layer0, 0, shared_resnets[0])

        # Convolute to 2D picture
        output = Conv2D(1, (1, 1), padding="same",
                        activation="sigmoid")(right0)

        model = keras.models.Model(input, output)
        return model
