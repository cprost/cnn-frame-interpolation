import predict

import pathlib

model_dir = pathlib.Path('model/checkpoints/')
input_data_dir = pathlib.Path('data/test/teapot_in.mp4')
temp_data_dir = pathlib.Path('tempfiles')
output_data_dir = pathlib.Path('video')

print('Generating an interpolated video from the trained model.')
src_width, src_height, src_fps = predict.generate_frames(input_data_dir, temp_data_dir)
predict.generate_video(model_dir, temp_data_dir, output_data_dir, src_fps, src_width, src_height)
