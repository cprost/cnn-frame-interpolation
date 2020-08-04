'''
Produces a higher-framerate video by producing and adding new interpolated
frames. Modified from data_processor.py

TODO: Refactor data_processor.py to include prediction-mode function
variants as used in this module
TODO: Add argparse functionality for prediction specs
'''

import parameters
from model import Interpolator

import shutil
import pathlib
import numpy as np
import cv2
import tensorflow as tf

def generate_frames(input_dir, output_dir):
    '''
    TODO: Refactor data_processor.py for inference mode
    '''

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    print('Processing video to {}'.format(output_dir))

    video_cap = cv2.VideoCapture(str(input_dir))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)

    print('Generating frames...')

    success, frame = video_cap.read()
    frame_num = 0
    while success:
        frame = cv2.resize(frame, (width, height))
        cv2.imwrite('{}/frame%04d.jpg'.format(output_dir) % frame_num, frame)
        success, frame = video_cap.read()  # read next frame
        frame_num += 1

    pair_count = 0

    frames = output_dir.glob('*')
    frames = sorted(frames)
    pair = []

    for frame in frames:
        pair.append(frame)

        if len(pair) == 2:

            # create triplet folder
            pathlib.Path('{}/{:04d}'.format(output_dir, pair_count)).mkdir(exist_ok=True)
            # move triplet
            for file in pair:
                shutil.copy(str(file), '{}/{:04d}/{}'.format(output_dir, pair_count, file.name))
            pair_count += 1
            pair.clear()
            pair.append(frame)

    return width, height, fps

def load_data(data_dir):
    dataset = tf.data.Dataset.list_files(str(data_dir / '*'), shuffle=False)  # grab list of all pair directories
    dataset = dataset.map(lambda dir: load_pair(dir))

    dataset = dataset.batch(parameters.BATCH_SIZE, drop_remainder=True)
        
    print("Dataset loaded!", str(data_dir))

    return dataset
    
def load_pair(pair_dir):
    frame_paths = tf.io.matching_files(pair_dir + '/*.jpg')
    frame_1 = load_frame(frame_paths[0])
    frame_2 = load_frame(frame_paths[1])

    return (frame_1, frame_2), frame_paths

def load_frame(frame_dir):
    frame = tf.io.read_file(frame_dir)
    frame = tf.image.decode_jpeg(frame)
    frame = tf.image.convert_image_dtype(frame, dtype=tf.float32) # no more 'frame / 255' to normalize

    return frame

def rgb_uint8(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)
    return image.astype(np.uint8)

def generate_video(model_dir, input_data_dir, output_data_dir, fps, width=parameters.IMG_WIDTH, height=parameters.IMG_HEIGHT):
    print('Predicting new frames for the interpolated output video...')
    model = Interpolator()
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    tf.train.Checkpoint(net=model).restore(ckpt_path)
    input_frames = load_data(input_data_dir)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(output_data_dir / 'skwarl3.mp4'), fourcc, fps*2, (width, height))

    for batch_num, batch in enumerate(input_frames):
        batch_frames, batch_paths = batch
        print(batch_num)
        frames_predict, motion = model(batch_frames, training=False)
        
        for i in range(parameters.BATCH_SIZE):
            out_video.write(rgb_uint8(batch_frames[0][i]))
            out_video.write(rgb_uint8(frames_predict[i]))

    out_video.release()

if __name__ == '__main__':
    model_dir = pathlib.Path('../model/checkpoints/')
    input_data_dir = pathlib.Path('../data/test/IMG_0045.mov')
    temp_data_dir = pathlib.Path('../tempfiles')
    output_data_dir = pathlib.Path('../video')

    width, height, fps_in = generate_frames(input_data_dir, temp_data_dir)
    generate_video(model_dir, input_data_dir, output_data_dir, fps_in, width, height)