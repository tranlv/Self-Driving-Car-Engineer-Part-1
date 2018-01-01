# **Advanced Lane Finding**

In this project, We attempted to identify the lane boundaries in a video.


The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Here are the [notebook](http://nbviewer.jupyter.org/gist/tranlyvu/ffb64be864e9b67cc2aa273d34df8b45) and [source code](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Advanced%20Lane%20Lines/src/advanced_lane_lines.py) of the project.

The project video is [here](https://youtu.be/6onP8z6ZPSI).

Writeup 
---

### Camera Calibration and distortion correction

The code for this step is contained in the 4th and 6th code cells of the IPython notebook located in "/notebook/advanced_lane_lines.ipynb" (or in lines 19 through 49 in /src/advanced_lane_lines.py).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![undistorted chessboard image](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Advanced%20Lane%20Lines/output_images/undistort_output.png)

### Pipeline (single images)

#### Distortion correction of a sample test image

In order to demonstrate form the piple to detect lane lines, I selected the first image in directory test_images

![sample image](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Advanced%20Lane%20Lines/test_images/test1.jpg)

Using Camera matrix and distortion coefficients from previous step, I apply distortion correction to this sample image

![undistort](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Advanced%20Lane%20Lines/output_images/undistort_test1.jpg)


#### Creating thresholded binary image

I used a combination of color and gradient thresholds to filter out what we don’t want.

```
1. First , we do a color threshold filter to pick only yelow and white color of the road lanes
```

I first visualize 3 color spaces RGB, HLS and HSV (color space transformation is done using cv2). Among all, R-channel in RGB , S-channel in HLS and V-channel in HSV seem to work best in their respective space. Their difference is very insignificant. I eventually decided to use value-channel of HSV, although I believe others will do equavalently well. 

With some error and trial by observation, I find the threshold (210, 255) to perform pretty well for  V-channel in HSV. 

![color filter](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Advanced%20Lane%20Lines/output_images/color_filter_binary.jpg)

Note that this can be done by seperating yellow and white color but i simplified the process by fiding the threshold for both white and yellow.

```
2. Next, I apply Sober operator (x and y), magnitude and direction of the gradient. After a few trial and error, I deduced the min and max threshold for each of them
```

```
3. The last thing is to combine threshold for both gradient and color
```

![gradient + color](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Advanced%20Lane%20Lines/output_images/combine_binary.jpg)

#### Perspective Transformation

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `src/advanced_lane_lines.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32([
    	[(img_size[0] / 4), 0],
    	[(img_size[0] / 4), img_size[1]],
    	[(img_size[0] * 3 / 4), img_size[1]],
    	[(img_size[0] * 3 / 4), 0]
    ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![wrap sample](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Advanced%20Lane%20Lines/output_images/warp_example_img.jpg)

#### Identifying lane-line pixels and fit their positions with a polynomial

I used second order polynomial to fit the lane: x = ay**2 + by + c.

In order to better estimate where the lane is, we use a histogram on the bottom half of image:

![histogram](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Advanced%20Lane%20Lines/output_images/histogram.jpg)

Then we divide the image in windows, and for each left and right window we find the mean of it, re-centering the window. The points inside the windows are stored. We then feed the numpy polyfit function to find the best second order polynomial to represent the lanes.


#### Radius of curvature of the Lane 

Radius of curvature is implemented from this [tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php). For second order polynomials, it is simplified to (1+(2Ay + B)**2)**1.5/ abs(2A).

I calculated both radius curvature in pixel and real world spaces, for example, here is radius curvature of the example image:

```
Pixel space: left curvature: 4826.07068768 , right curvature: 2966.83010737
World space: left curvature: 2003.07070208 m , right curvature: 1745.93214794 m
```

Here is the final output of the test image number 1 where lane area is identified clearly:

![final putput](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Advanced%20Lane%20Lines/output_images/test_img_1_output.jpg)

---

### Pipeline (video)

Here's a [link to my video result](https://youtu.be/6onP8z6ZPSI) using the model above.

Udacity recommended using python classes to keep track of the line detected between frames, however i finally use 'global' variable to do so.

---

### Discussion

One improvement that has been discussed on forum is to average lane lines over a few frames. However, since the video shows lane lines detected quite clearly for project video I did not try to implement it, maybe I can try for challenge video.
