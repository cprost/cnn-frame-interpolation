import tensorflow as tf

# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.unet = UNet(4)
        


class UNet(tf.keras.layers.Layer):
    # instantiates the UNet parent class
    def __init__(self, name="UNet"):
        super(UNet, self).__init__(name=name)

    def build(self, input_shape):
        self.conv_1_1 = ConvLayer(32, 3)
        self.conv_1_2 = ConvLayer(32, 3)
        self.maxpool_1 = MaxPool(2)
        self.conv_2_1 = ConvLayer(64, 3)
        self.conv_2_2 = ConvLayer(64, 3)
        self.maxpool_2 = MaxPool(2)
        self.conv_3_1 = ConvLayer(128, 3)
        self.conv_3_2 = ConvLayer(128, 3)
        self.maxpool_3 = MaxPool(2)
        self.conv_4_1 = ConvLayer(256, 3)
        self.conv_4_2 = ConvLayer(256, 3)
        self.maxpool_4 = MaxPool(2)
        self.conv_5_1 = ConvLayer(512, 3)
        self.conv_5_2 = ConvLayer(512, 3)

        self.upmerge_6 = UpMerge(256, 3)
        self.conv_6_1 = ConvLayer(256, 3)
        self.conv_6_2 = ConvLayer(256, 3)
        self.upmerge_7 = UpMerge(128, 3)
        self.conv_7_1 = ConvLayer(128, 3)
        self.conv_7_2 = ConvLayer(128, 3)
        self.upmerge_8 = UpMerge(64, 3)
        self.conv_8_1 = ConvLayer(64, 3)
        self.conv_8_2 = ConvLayer(64, 3)
        self.upmerge_9 = UpMerge(32, 3)
        self.conv_9_1 = ConvLayer(32, 3)
        self.conv_9_2 = ConvLayer(32, 3)
        self.output = tf.keras.layers.Conv2D(
            kernel_size=3,
            strides=(1, 1),
            activation='relu',
            padding='same'
        )
    
    def call(self, inputs):
        cd_1_1 = self.conv_1_1(inputs)
        cd_1_2 = self.conv_1_2(cd_1_1)
        mp_1 = self.maxpool_1(cd_1_2)
        cd_2_1 = self.conv_2_1(mp_1)
        cd_2_2 = self.conv_2_2(cd_2_1)
        mp_2 = self.maxpool_2(cd_2_2)
        cd_3_1 = self.conv_3_1(mp_2)
        cd_3_2 = self.conv_3_2(cd_3_1)
        mp_3 = self.maxpool_3(cd_3_2)
        cd_4_1 = self.conv_4_1(mp_3)
        cd_4_2 = self.conv_4_2(cd_4_1)
        mp_4 = self.maxpool_4(cd_4_2)
        cd_5_1 = self.conv_5_1(mp_4)
        cd_5_2 = self.conv_5_2(cd_5_1)
        um_6 = self.upmerge_6([cd_5_2, cd_4_2])
        cu_6_1 = self.conv_6_1(um_6)
        cu_6_2 = self.conv_6_2(cu_6_1)
        um_7 = self.upmerge_7([cu_6_2, cd_3_2])
        cu_7_1 = self.conv_7_1(um_7)
        cu_7_2 = self.conv_7_2(cu_7_1)
        um_8 = self.upmerge_8([cu_7_2, cd_2_2])
        cu_8_1 = self.conv_8_1(um_8)
        cu_8_2 = self.conv_8_2(cu_8_1)
        um_9 = self.upmerge_9([cu_8_2, cd_1_2])
        cu_9_1 = self.conv_9_1(um_9)
        cu_9_2 = self.conv_9_2(cu_9_1)
        unet = self.output(cu_9_2)

        return unet

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv_layer = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation='relu',
            padding='same'
        )

    def call(self, inputs):
        net = self.conv_layer(inputs)
        return net

class MaxPool(tf.keras.layers.Layer):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def build(self, input_shape):
        self.max_pool = tf.keras.layers.MaxPool2D(
            pool_size=self.pool_size
        )

    def call(self, inputs):
        net = self.max_pool(inputs)
        return net


class UpMerge(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.upsample = tf.keras.layers.UpSampling2D(
            size=(2, 2)
        )

    def call(self, inputs):
        prev_layer, down_layer = inputs
        up_layer = self.upsample(prev_layer)
        net = tf.keras.layers.concatenate([up_layer, down_layer])
        return net