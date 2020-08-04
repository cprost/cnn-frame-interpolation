import data_processor
from model import Interpolator
import parameters
import losses

import pathlib
import tensorflow as tf

def train():
    '''
    loads batches of training data (triplets) and trains the CNN over various epochs
    saves the model in checkpoints to allow pauses to training. Similar to predict.py

    TODO: add validation
    TODO: add argparse functionality
    '''

    # load training data
    train_data_dir = pathlib.Path('../out')
    train_data = data_processor.load(train_data_dir, training=True)
    train_data_batches = tf.data.experimental.cardinality(train_data).numpy()

    print('{} batches of {} triplets each'.format(train_data_batches, parameters.BATCH_SIZE))

    # prepare directories for the model
    model_dir = pathlib.Path('../model')
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / 'checkpoints'

    # load the model and any checkpoints, if they exist
    model = Interpolator()
    optimizer = tf.keras.optimizers.Adam(lr=parameters.ADAM_LR)
    loss_func = losses.Loss()
    iterator = iter(train_data)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    progress_bar = tf.keras.utils.Progbar(train_data_batches)
    
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print('Restored latest checkpoint!')

    for epoch in range(parameters.EPOCHS):
        print('Epoch: {}'.format(epoch))

        for batch_num, batch in enumerate(train_data):
            loss, ssim, psnr = train_batch(model, batch, optimizer, loss_func)
            # loss = total_loss, r_loss, p_loss, w_loss, s_loss

            progress_vals = [
                ('Total loss', loss[0]),
                ('SSIM', ssim),
                ('Rec. loss', loss[1]),
                ('Perc. loss', loss[2]),
                ('Warp. loss', loss[3]),
                ('Smooth. loss', loss[4])
            ]
            
            progress_bar.update(batch_num + 1, progress_vals)
        
        checkpoint.save(pathlib.PurePath(checkpoint_dir,  'checkpoint'))
        print('Created checkpoint for epoch #{}!'.format(epoch))

    print('Training done!')

@tf.function
def train_batch(model, batch, optimizer, loss_func):
    '''
    This function and all children will produce a dataflow graph for faster
    performance than in eager mode (default). Disabling @tf.function will allow
    for easier debugging at the cost of performance.
    '''

    f_1, f_2, f_3 = batch  # f_1 and f_3 are input frames, f_2 is the target frame
    with tf.GradientTape() as tape:
        prediction, prediction_motion = model((f_1, f_3), f_2)
        loss = loss_func.total_loss((f_1, f_3), f_2, prediction, prediction_motion)
        ssim = loss_func.ssim(prediction, f_2)
        psnr = loss_func.psnr(prediction, f_2)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, tf.math.reduce_mean(ssim), tf.math.reduce_mean(psnr)

if __name__ == '__main__':
    train()