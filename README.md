## Video smoothing CNN

This is a TensorFlow-based convolutional neural network (CNN) that generates interpolated video frames from two sequential input frames. It uses a UNet-inspired CNN architecture for segmenting components of the input frames that may move at different rates, yielding smooth results with fewer artifacts versus conventional "naïve" methods.

Image segmentation can identify objects that become hidden or revealed between frames, allowing the CNN to effectively interpret the object's intermediate position between frames. Traditional interpolation methods create ghost objects which can lead to sharp decreases in visual quality, particularly during high-motion scenes in videos. 


Car video - interpolated second frame

![Interpolated animation of car](/results/0_interp.gif)

Car video - true second frame

![True animation of car](/results/0_true.gif)

Wedding video - interpolated second frame

![Interpolated animation of wedding](/results/1_interp.gif)

Wedding video - true second frame

![True animation of wedding](/results/1_true.gif)
