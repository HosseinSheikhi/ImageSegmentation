import tensorflow as tf


class VggBlockWithBN(tf.keras.layers.Layer):
    def __init__(self, layers, filters, kernel_size, stride=1):
        super(VggBlockWithBN, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = stride
        self.layers = layers
        conv_layers = [tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.stride, padding="SAME",
                                              kernel_initializer='he_normal') for i in range(self.layers)]
        bn_layers = [tf.keras.layers.BatchNormalization() for i in range(self.layers)]
        self.conv_bn = zip(conv_layers, bn_layers)

    def call(self, inputs, training):
        x = inputs
        for conv, bn in self.conv_bn:
            x = conv(x)
            x = bn(x, trainable=training)
            x = tf.nn.relu(x)
