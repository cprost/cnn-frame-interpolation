# Callbacks and checkpoints for saving WIP models during training and restoring trained models

import keras

checkpoint_path = './weights/weights.hdf5'

cb_checkpoint = keras.callbacks.ModelCheckpoint(monitor='loss',
                                                filepath=checkpoint_path,
                                                verbose=1,
                                                save_best_only=True)

# reducing learning rate necessary when no loss reduction is made after 3 epochs
cb_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                 factor=0.5,
                                                 patience=3,
                                                 min_lr=0.0001)
