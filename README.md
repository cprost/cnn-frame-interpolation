# Video smoothing CNN

![Header](/images/header.jpg)

### Background

This is a TensorFlow-based convolutional neural network (CNN) that generates interpolated video frames from two sequential input frames. It uses a UNet-inspired CNN architecture for segmenting components of the input frames that may move at different rates, yielding smooth results with fewer artifacts versus conventional "naïve" methods.

<img src="https://github.com/cprost/cnn-frame-interpolation/blob/master/images/interp_vector.png" width="550">

Image segmentation can identify objects that become hidden or revealed between frames, allowing the CNN to effectively interpret the object's intermediate position between frames. Traditional interpolation methods create ghost objects which can lead to sharp decreases in visual quality, particularly during high-motion scenes in videos. 

<img src="https://github.com/cprost/cnn-frame-interpolation/blob/master/images/interp_occlude.png" width="550">

<img src="https://github.com/cprost/cnn-frame-interpolation/blob/master/images/occlusion.png" width="550">

### Implementation

This model was trained using the [Vimeo90k dataset](http://toflow.csail.mit.edu/), containing several thousand video *triplets* - sets of three sequential frames, where the second frame is used as the ground truth frame for training the CNN. Three different loss functions (mean squared error, Charbonnier, and the structural similarity index) were used to train the CNN for 100 epochs. Training was performed on a hardware-accelerated Google Cloud instance, using mini-batch gradient descent.

Note: Due to I/O limitations, a smaller subset of the Vimeo90k dataset was used. Further training with additional triplets and/or training epochs is recommended, if access to unrestricted hardware is available.

### Results

![Comparison of results and loss functions](/images/comparison.png)

Car video - interpolated second frame

![Interpolated animation of car](/results/0_interp.gif)

Car video - true second frame

![True animation of car](/results/0_true.gif)

Wedding video - interpolated second frame

![Interpolated animation of wedding](/results/1_interp.gif)

Wedding video - true second frame

![True animation of wedding](/results/1_true.gif)
