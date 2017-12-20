# **Finding Lane Lines on the Road** 

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

The project objective is to detect lane lines in images using Python and OpenCV. 



Writeup 
---

[//]: # (Image References)

[image1]: test_images_output/grayscale.jpg "Grayscale"

---

### 1. Pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

First, I retained all white lane lines

![alt text][image1]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 



### 2. Identify potential shortcomings with your current pipeline




### 3. Suggest possible improvements to your pipeline



