import tensorflow as tf
from VGG16.VggBlock import VggBlock


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv_blk_5 = None
        self.conv_blk_4 = None
        self.conv_blk_3 = None
        self.conv_blk_2 = None
        self.conv_blk_1 = None

        self.trans_conv_blk_1 = None
        self.trans_conv_blk_2 = None
        self.trans_conv_blk_3 = None
        self.trans_conv_blk_4 = None
        self.trans_conv_blk_5 = None

    def build(self, input_shape):
        self.conv_blk_5 = VggBlock(layers=3, filters=512, kernel_size=3, name="dec_conv_blk5")
        self.trans_conv_blk_5 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding="same")
        self.conv_blk_4 = VggBlock(layers=3, filters=512, kernel_size=3, name="dec_conv_blk4")
        self.trans_conv_blk_4 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding="same")
        self.conv_blk_3 = VggBlock(layers=3, filters=256, kernel_size=3, name="dec_conv_blk3")
        self.trans_conv_blk_3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding="same")
        self.conv_blk_2 = VggBlock(layers=2, filters=128, kernel_size=3, name="dec_conv_blk2")
        self.trans_conv_blk_2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same")
        self.conv_blk_1 = VggBlock(layers=2, filters=64, kernel_size=3, name="dec_conv_blk1")
        self.trans_conv_blk_1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same")

    def call(self, inputs, blk_1_out, blk_2_out, blk_3_out, blk_4_out, blk_5_out):
        x = self.trans_conv_blk_5(inputs)
        x = tf.keras.layers.concatenate([blk_5_out, x])
        x = self.conv_blk_5(x)

        x = self.trans_conv_blk_4(x)
        x = tf.keras.layers.concatenate([blk_4_out, x])
        x = self.conv_blk_4(x)

        x = self.trans_conv_blk_3(x)
        x = tf.keras.layers.concatenate([blk_3_out, x])
        x = self.conv_blk_3(x)

        x = self.trans_conv_blk_2(x)
        x = tf.keras.layers.concatenate([blk_2_out, x])
        x = self.conv_blk_2(x)

        x = self.trans_conv_blk_1(x)
        x = tf.keras.layers.concatenate([blk_1_out, x])
        x = self.conv_blk_1(x)

        return x
