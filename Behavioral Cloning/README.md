# Behaviorial Cloning 

Overview
---

In this project, we will use deep neural networks and convolutional neural networks to clone driving behavior. The model will output a steering angle to an autonomous vehicle.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

This is [notebook]() and source code()


Project Writeup
---

### Files Submitted & Code Quality

My project includes the following files:

```
- src/model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network 
```

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.


Video can be created using commands:

```
python drive.py model.h5 run1
python video.py run1

```
run1 is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

Second command creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.


Optionally, one can specify the FPS (frames per second) of the video:

```
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Dataset

I used training dataset provided by Udacity. I use all 3 positions of camera with correction of 0.25 , i.e addition of 0.25 to steering angle for left-positioned camera and substraction of 0.25 for right-positioned camera.

I could have self-produced ore data but due to time constraint, I only used Udacity dataset

Moreover, after unable to complete a whole lap, I searched the forum and finally decided to randomly choose camera to select from

The dataset is split into 20% of test set. Also, the training set is shuffled before training



### Data Preprocessing

My Pre-processing pipeline

```
- Data augmentation: Fliping the image horizontal randomly 
- Cropping the image to 66x200 to fit NVIDIA model
- Normalization and Mean centering
```

Again, Flipping the image randomly was recommended by forum mentor.


### Model Architecture and Training Strategy

In my first attempt, I used 9-layers network from end to end learning for self-driving cars by NVIDIA as recommended by Udacity

    
|Layer   |type    |output filter/neurons|
|--------|--------|--------|
|1       |conv    |24      |
|2       |conv    |36      |
|3       |conv    |48      |
|4       |conv    |64      |
|5       |conv    |64      |
|6       |flattern|1164    |
|7       |relu    |100     |
|8       |relu    |50      |
|9       |relu    |10      |
|10      |relu    |1       |


However, I detected overfitting in my first attempt, and hence i tried to improved the mode in second model by using regulation, i.e dropout

|Layer   |type    |output filter/neurons|
|--------|--------|--------|
|1       |conv    |24      |
|        |dropout |        |
|2       |conv    |36      |
|        |dropout |        |
|3       |conv    |48      |
|        |dropout |        |
|4       |conv    |64      |
|5       |conv    |64      |
|6       |flattern|1164    |
|7       |relu    |100     |
|8       |relu    |50      |
|9       |relu    |10      |
|10      |relu    |1       |


The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Final Model parameters: 

```
- Optimizer: Adam optimizer, so the learning rate was not tuned manually 
- Epoch: 5
- Batch size: 32
```

