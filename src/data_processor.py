'''
This is a standalone file for creating datasets and lists of files
Datasets available from:
http://toflow.csail.mit.edu/
http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/
'''

import parameters

import pathlib
import shutil
import cv2
import tensorflow as tf

def load(data_dir, training=True):
    '''
    TODO: add dataset sharding for systems where RAM < dataset total size
    '''

    dataset = tf.data.Dataset.list_files(str(data_dir / '*'))  # grab list of all triplet directories
    dataset = dataset.map(lambda dir: load_triplet(dir))

    if training:
        dataset = dataset.shuffle(parameters.SHUFFLE_BUF)  # DO NOT SHUFFLE ON INFERENCE!!!

    dataset = dataset.batch(parameters.BATCH_SIZE, drop_remainder=True)

    # sanity check
    # for batch_num, batch in enumerate(dataset):
    #     print('Batch:', batch_num)
    #     f1, f2, f3 = batch
    #     triplets = zip(f1, f2, f3)

    #     for i, triplet in enumerate(triplets):
    #         print(i)
    #         print(triplet[0].shape)
        
    print("Dataset loaded!", str(data_dir))

    return dataset

def load_triplet(triplet_dir):
    '''
    TODO: add per-triplet data augmentation
    '''

    frame_paths = tf.io.matching_files(triplet_dir + '/*.jpg')
    frame_1 = load_frame(frame_paths[0])
    frame_2 = load_frame(frame_paths[1])  # this is the target frame
    frame_3 = load_frame(frame_paths[2])

    return (frame_1, frame_2, frame_3)

def load_frame(frame_dir):
    frame = tf.io.read_file(frame_dir)
    frame = tf.image.decode_jpeg(frame)
    frame = tf.image.convert_image_dtype(frame, dtype=tf.float32) # no more 'frame / 255' to normalize

    return frame

def generate_frames(input_dir, output_dir, width=1280, height=720):
    print('Generating frames...')

    # extract frames from video into temp folders
    folder = input_dir.glob('**/*')
    total_videos = len(list(folder))
    video_num = 1

    for video_name in input_dir.glob('**/*'):
        output_subdir = (output_dir / ('temp_' + video_name.name)).with_suffix('')
        print('Processed file {}/{} to {}'.format(video_num, total_videos, output_subdir))
        pathlib.Path(output_subdir).mkdir(parents=True, exist_ok=True)

        video_cap = cv2.VideoCapture(str(video_name))
        success, frame = video_cap.read()
        frame_num = 0
        while success:
            frame = cv2.resize(frame, (width, height))
            cv2.imwrite('{}/frame%04d.jpg'.format(output_subdir) % frame_num, frame)
            success, frame = video_cap.read()  # read next frame
            frame_num += 1

        video_num += 1

    temp_folders = list(output_dir.glob('*'))

    # generate training triplets
    triplet_count = 0

    for temp_folder in temp_folders:
        frames = temp_folder.glob('*')
        frames = sorted(frames)
        triplet = []

        for frame in frames:
            triplet.append(frame)

            if len(triplet) == 3:

                # create triplet folder
                pathlib.Path('{}/{}'.format(output_dir, triplet_count)).mkdir(exist_ok=True)
                # move triplet
                for file in triplet:
                    shutil.move(str(file), '{}/{}/{}'.format(output_dir, triplet_count, file.name))
                triplet_count += 1
                triplet.clear()

    # remove temp folders and any incomplete triplets
    for temp_folder in temp_folders:
        files = temp_folder.glob('*')

        for file in files:
            file.unlink()

        temp_folder.rmdir()

if __name__ == '__main__':
    input_data_dir = pathlib.Path('../data/train')
    output_data_dir = pathlib.Path('../out')
    generate_frames(input_data_dir, output_data_dir, width=parameters.IMG_WIDTH, height=parameters.IMG_HEIGHT)