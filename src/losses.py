'''
This defines custom loss functions for training the interpolator, and ancillary
metrics like the structural similary index (SSIM) and peak signal-noise ratio
(PSNR) for evaluating the interpolator's performance.

Calling the total_loss method in train.py will execute all loss-related functions
in graph mode, remove the @tf.function line in train.py to facilitate debugging
these functions.
'''

import parameters
import tensorflow as tf

class Loss:
    def __init__(self):
        '''
        Uses a pre-trained VGG16 network for implementing perceptual loss, as
        first suggested by Jiang et al. for Super SloMo
        https://stackoverflow.com/a/46359250
        '''

        vgg = tf.keras.applications.VGG16(include_top=False)
        outputs = vgg.get_layer('block4_conv3').output
        self.vgg = tf.keras.Model(inputs=vgg.inputs, outputs=outputs, trainable=False)

        # instantiate loss objects beforehand, no eager computation
        # conveniently reduces to scalar as reduction=auto, unlike tf.keras.losses.MSE()/MAE()
        self.mse = tf.keras.losses.MeanSquaredError()  # L2
        self.mae = tf.keras.losses.MeanAbsoluteError()  # L1

    def reconstruction_loss(self, frame_target, frame_predicted):
        return self.mae(frame_target, frame_predicted)

    def perceptual_loss(self, frame_target, frame_predicted):
        # pass each frame through vgg16 to extract sharpening features for true and interp frames

        for layer in self.vgg.layers:
            frame_target = layer(frame_target)
            frame_predicted = layer(frame_predicted)
        
        return self.mse(frame_target, frame_predicted)

    def warping_loss(self, frame_1, frame_3, frame_2, frames_input_warped, frames_target_warped):

        loss_1 = self.mae(frame_1, frames_input_warped[1])
        loss_2 = self.mae(frame_3, frames_input_warped[0])
        loss_3 = self.mae(frame_2, frames_target_warped[0])
        loss_4 = self.mae(frame_2, frames_target_warped[1])
        
        return loss_1 + loss_2 + loss_3 + loss_4

    def smoothness_loss(self, flow_1_3, flow_3_1):
        # compute del_x, del_y by a 1-pixel shift of the input flows
        
        del_1_3_x = tf.reduce_mean(tf.abs(flow_1_3[:, 1:, :, :] - flow_1_3[:, :-1, :, :]))
        del_1_3_y = tf.reduce_mean(tf.abs(flow_1_3[:, :, 1:, :] - flow_1_3[:, :, :-1, :]))

        del_3_1_x = tf.reduce_mean(tf.abs(flow_3_1[:, 1:, :, :] - flow_3_1[:, :-1, :, :]))
        del_3_1_y = tf.reduce_mean(tf.abs(flow_3_1[:, :, 1:, :] - flow_3_1[:, :, :-1, :]))

        del_1_3 = del_1_3_x + del_1_3_y
        del_3_1 = del_3_1_x + del_3_1_y

        return del_1_3 + del_3_1

    def total_loss(self, frames_input, frame_target, frame_predicted, motion):
        frame_1, frame_3 = frames_input

        flow_1_3, flow_3_1 = motion[:2]
        frames_input_warped = motion[2:4]  # frames (1,3) created from warping each other's flows
        frames_target_warped = motion[4:]  # target frame created from warping (1,3)'s flows

        r_loss = self.reconstruction_loss(frame_target, frame_predicted)
        p_loss = self.perceptual_loss(frame_target, frame_predicted)
        w_loss = self.warping_loss(frame_1, frame_3, frame_target, frames_input_warped, frames_target_warped)
        s_loss = self.smoothness_loss(flow_1_3, flow_3_1)

        total_loss = (
            r_loss * parameters.LAMBDA_R
            + p_loss * parameters.LAMBDA_P
            + w_loss * parameters.LAMBDA_W
            + s_loss * parameters.LAMBDA_S
        )

        return total_loss, r_loss, p_loss, w_loss, s_loss  # loss_components
   
    def ssim(self, frame_target, frame_predicted):
        return tf.image.ssim(frame_target, frame_predicted, max_val=1.0)

    def psnr(self, frame_target, frame_predicted):
        return tf.image.psnr(frame_target, frame_predicted, max_val=1.0)