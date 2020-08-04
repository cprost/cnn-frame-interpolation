# Video smoothing CNN

![Header comparison gif - city](/results/city_comparison.gif)

### Background

This is a TensorFlow-based convolutional neural network (CNN) that generates interpolated video frames from two sequential input frames. It uses a UNet-based CNN architecture for computing the flow of segmented objects between frames, yielding smooth results with fewer artifacts versus conventional "naïve" methods like bilinear interpolation.

<img src="https://github.com/cprost/cnn-frame-interpolation/blob/master/images/interp_vector.png" width="550">

Segmentation in conjunction with flow computation can identify objects that become hidden or revealed between frames, allowing the CNN to effectively interpret the object's intermediate position between frames. Traditional interpolation methods create ghost objects which can lead to sharp decreases in visual quality, particularly during high-motion scenes in videos. 

<img src="https://github.com/cprost/cnn-frame-interpolation/blob/master/images/interp_occlude.png" width="550">

### Implementation

The implemented model was trained by providing a sequential frame *triplet* for each training step: *Frames 1* and *3* were provided as input, with the intermediate *frame #2* serving as the training target. Each step produced a synthetic *frame t* to be compared against the target *frame #2*, by means of a composite loss function.

### Results

*Note: the .gifs may demonstrate aliasing, depending on your screen size. View the full-size images for better quality.*

![Drone comparison](/results/drone_comparison.gif)

![Tea pouring comparison](/results/tea_comparison.gif)
