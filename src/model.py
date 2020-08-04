import tensorflow as tf
import tensorflow_addons as tfa

# https://arxiv.org/pdf/1712.00080.pdf
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
# https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_training_argument_in_the_call_method

class Interpolator(tf.keras.Model):
    '''
    Calling the model will interpolate the flow to an intermediate frame,
    generate this frame based on the interpolated flow, and return the frame
    along with the flow data for loss calculations during training.
    '''

    def __init__(self):
        super().__init__()
        self.unet = UNet(filters=4)  # two sets of 2-channel flows
        self.flow_interpolation = FlowInterpolation()

    def call(self, frames_input, frame_target=None):
        frame_1, frame_3 = frames_input

        # calculate flow between input frames 1 and 3
        flow_computed = self.unet(tf.concat([frame_1, frame_3], axis=-1))
        flow_1_3 = flow_computed[:, :, :, 2:]  # flow from 1 to 3
        flow_3_1 = flow_computed[:, :, :, :2]  # flow from 3 to 1

        # interpolate flow from frames 1,3 above to the intermediate frame 2
        flow_interp_in = (frame_1, frame_3, flow_1_3, flow_3_1)
        flow_interp_out = self.flow_interpolation(flow_interp_in)
        frame_2_1, frame_2_3, flow_1_2, flow_3_2, frame_vis_1_2, frame_vis_3_2 = flow_interp_out

        # create warped frames (1,3) from each other, use for computing loss
        frame_1_3 = warp_frame(frame_3, flow_3_1)
        frame_3_1 = warp_frame(frame_1, flow_1_3)

        # use flow and warped frames to predict the target intermediate frame
        prediction = predict(frame_1, frame_3, flow_1_2, flow_3_2, frame_vis_1_2, frame_vis_3_2)
        prediction_motion = (flow_1_3, flow_3_1, frame_1_3, frame_3_1, frame_2_1, frame_2_3)

        return prediction, prediction_motion

class UNet(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters

    def build(self, input_shape):
        # defines layer filter counts and kernel sizes, to then be connected in call()
        # scaling down the largest layers from 512 (original UNet) to 384 for RAM savings

        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same')  # call relu after
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same')  # call relu after
        self.down_1 = Down(32, 5)
        self.down_2 = Down(64, 5)
        self.down_3 = Down(128, 3)
        self.down_4 = Down(256, 3)
        self.down_5 = Down(384, 3)
        self.up_1 = UpMerge(384, 3)
        self.up_2 = UpMerge(384, 3)
        self.up_3 = UpMerge(256, 3)
        self.up_4 = UpMerge(128, 3)
        self.up_5 = UpMerge(64, 3)
        self.conv_out = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, padding='same')
    
    def call(self, input):
        # skipcc are copied and cropped skip connections used as inputs for upsampling + merging

        x = self.conv_1(input)
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        ccskip_1 = self.leaky_relu(x)
        ccskip_2 = self.down_1(ccskip_1)
        ccskip_3 = self.down_2(ccskip_2)
        ccskip_4 = self.down_3(ccskip_3)
        ccskip_5 = self.down_4(ccskip_4)
        x = self.down_5(ccskip_5)
        x = self.up_1(x, ccskip_5)
        x = self.up_2(x, ccskip_4)
        x = self.up_3(x, ccskip_3)
        x = self.up_4(x, ccskip_2)
        x = self.up_5(x, ccskip_1)
        x = self.conv_out(x)
        x = self.leaky_relu(x)

        return x

class Down(tf.keras.layers.Layer):
    # Downsampling and convolving unit for feature extraction
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same')
        self.pool = tf.keras.layers.AveragePooling2D()  # smoother feat extraction vs max-pool
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, input):
        x = self.pool(input)
        x = self.conv_1(x)
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        x = self.leaky_relu(x)
        return x


class UpMerge(tf.keras.layers.Layer):
    # Upsampling layer unit that takes merged outputs from prior layers as input
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same')
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.cat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input, ccskip):
        x = self.upsample(input)
        x = self.conv_1(x)
        x = self.leaky_relu(x)

        # the feed-forward layer 'x' may be smaller than the skip layer, so pad x to the same size
        dim_1 = max(ccskip.shape[1], x.shape[1]) - min(ccskip.shape[1], x.shape[1])
        dim_2 = max(ccskip.shape[2], x.shape[2]) - min(ccskip.shape[2], x.shape[2])
        paddings = tf.constant([[0, 0], [dim_1, 0], [dim_2, 0], [0, 0]])

        x = tf.pad(x, paddings=paddings)

        x = self.cat([x, ccskip])
        x = self.conv_2(x)
        x = self.leaky_relu(x)
        return x

class FlowInterpolation(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.unet = UNet(filters=5)  # two sets of two-channel flow, and visibility map

    def call(self, inputs):
        frame_1, frame_3, flow_1_3, flow_3_1 = inputs

        flow_1_2 = 0.5 * flow_1_3
        flow_3_2 = 0.5 * flow_3_1

        frame_2_1 = warp_frame(frame_1, flow_1_2)
        frame_2_3 = warp_frame(frame_3, flow_3_2)

        interp_input = [frame_1, frame_3, flow_1_3, flow_3_1, flow_1_2, flow_3_2, frame_2_1, frame_2_3]
        interp_input = tf.concat(interp_input, axis=-1)
        flow_maps = self.unet(interp_input)

        # optical flow residuals and visibility maps
        delta_flow_2_1 = flow_maps[:, :, :, :2]
        delta_flow_2_3 = flow_maps[:, :, :, 2:4]

        # soft visibility map
        frame_vis_1_2 = tf.math.sigmoid(flow_maps[:, :, :, 4:])
        frame_vis_3_2 = 1 - frame_vis_1_2

        flow_1_2 = flow_1_2 + delta_flow_2_1
        flow_3_2 = flow_3_2 + delta_flow_2_3

        return frame_2_1, frame_2_3, flow_1_2, flow_3_2, frame_vis_1_2, frame_vis_3_2

def warp_frame(input_frame, input_flow):
    return tfa.image.dense_image_warp(input_frame, input_flow)

def predict(frame_1, frame_3, flow_1_2, flow_3_2, frame_vis_1_2, frame_vis_3_2):
    '''
    Uses adjusted flow and visibility maps from FlowInterpolation to predict
    an intermediate frame_2 between original frames 1 and 3. This can then be
    compared against the ground-truth frame for training the network, or to
    produce a frame-interpolated video during the inference process
    '''

    # compute an updated intermediate frame 2
    frame_2_1 = warp_frame(frame_1, flow_1_2)
    frame_2_3 = warp_frame(frame_3, flow_3_2)

    frame_interp = 0.5 * (frame_2_1 * frame_vis_1_2 + frame_2_3 * frame_vis_3_2)
    adjust = 0.5 * (frame_vis_1_2 + frame_vis_3_2)  # for adjusting the visibility of each pixel
    return tf.divide(frame_interp, adjust)