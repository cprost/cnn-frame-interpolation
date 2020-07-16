import tensorflow as tf

class Loss:
    def __init__(self):
        self.mse = tf.keras.losses.MeanSquaredError()

    def calc_loss(self, frame_1, frame_3, frame_target, frame_int):
        return self.mse(frame_target, frame_int)