## Traffic Sign Classification

Overview
---

In this project, we will use deep neural networks and convolutional neural networks to classify traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) as sample dataset. 
 
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here are the [notebook](http://nbviewer.jupyter.org/gist/tranlyvu/83ae4a2ef68908f33b3c4f3d11b1e374) and [source code](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Traffic%20Sign%20Classifier/src/second_attempt.py) of this project.

Project writeup
---

### Data Set Summary & Exploration

#### Summary of the dataset

Udacity provided the German Traffic Sign Dataset in pickled form which is loaded using Python package 'pickle'

The provided dataset has 3 separate sets: training, validation and test sets, hence i do not have to split data for validation purpose. Here are some information:
```
The size of training set is 34799
The size of the validation set is 4410
The size of test set is 12630
The shape of a traffic sign image is (32, 32, 3)
The number of unique classes/labels in the data set is 43 
```

The shape of traffic sign implies 32x32 pixels image with 3 channels, this is because Udacity has resized the images them before providing to students.

#### Exploratory visualization of the dataset

I did not spend so much time on this. I first print out the distribution of the samples in 43 classes of labels which 'Speed limit (50km/h)' sign has most samples (2010 samples) following by 'Speed limit (30km/h)' sign (1980 samples) and  'Yield' sign (1920 samples).

I have also plotted out 10 random images which can be seen in notebook.

### Design and Test a Model Architecture

#### Image data pre-processing

As a first step, I decided to convert the images to grayscale to convert to 1 channel image and remove the effect of color. 

Next, I normalized normalized the data so that the data has mean zero and equal variance, i.e (pixel - 128.0)/ 128.0

Here is an example of an image after preprocessing.

![pre-processed](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Traffic%20Sign%20Classifier/test_images_output/preprocessed_img.jpg)

#### Model architecture and training

My first attempt was to try the famous Lenet-5 model as recommended by Udacity because convolutional model is considered to performed best on object recognition:


|Layer   |type    |Input   |output  |
|--------|--------|--------|--------|
|1       |conv    |32x32x1 |28x28x6 |
|        |relu    |        |        |
|        |max_pool|28x28x6 |14x14x6 |
|2       |conv    |14x14x6 |10x10x16|
|        |relu    |        |        |
|        |max_pool|10x10x16|5x5x16  |
|        |flatten |5x5x16  |400     |
|3       |linear  |400     |120     |
|        |relu    |        |        |
|4       |linear  |120     |84      |
|        |relu    |        |        |
|5       |linear  |84      |43      |


First attempt only gave me 86% vadilation accuracy with 28 epochs. Validation loss is way higher than training loss and they converge at different values. This is strong signal of overfitting.

There are few techniques to battle overfitting:

```
- Increase training dataset
- Regulazation, i.e dropout
- Reduce the complexity of training model
```

The conplexity of original Lener-5 is pretty simple, so I chose to apply dropout of 0.5 to every layers of the model. After running for 300 epochs, my validation accuracy reached 89% and there is no signal of overfitting. I decided to increase the complexity of model to improve the accuracy.

This is my final model architecture

|Layer   |type    |Input   |output  |
|--------|--------|--------|--------|
|1       |conv    |32x32x1 |28x28x10|
|        |relu    |        |        |
|        |dropout |        |        |
|2       |conv    |28x28x10|24x24x20|
|        |relu    |        |        |
|        |dropout |        |        |
|3       |conv    |24x24x10|20x20x30|
|        |relu    |        |        |
|        |dropout |        |        |
|4       |conv    |20x20x30|16x16x40|
|        |relu    |        |        |
|        |max_pool|16x16x40|8x8x40  |
|        |dropout |        |        |
|        |flatten |8x8x40  |2560    |
|5       |linear  |2560    |1280    |
|        |relu    |        |        |
|6       |linear  |1280    |640     |
|        |relu    |        |        |
|7       |linear  |640     |320     |
|        |relu    |        |        |
|8       |linear  |320     |160     |
|        |relu    |        |        |
|9       |linear  |160     |80      |
|        |relu    |        |        |
|10      |linear  |80      |43      |


Here are some information of my model training

``` 
- Type of optimizer: Adam Optimizer (which is generally considered the best)
- The batch size: 128
- Nnumber of epochs: 40 
- learning rate: 0.0001


```
My final model results were:

```
- training set accuracy of: 0.998%
- validation set accuracy of: 0.972%
- The training and validation loss converged at around < 0.12
- test set accuracy: 0.95

```

Note that the dropout of 0.5 applied only during training phase


###Test a Model on New Images

#### Testing on new five German traffic signs

Here are five German traffic signs that I found on the web:

![img1](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Traffic%20Sign%20Classifier/new_images/stop_sign.jpg) ![img2](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Traffic%20Sign%20Classifier/new_images/yield_sign.jpg) ![img3](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Traffic%20Sign%20Classifier/new_images/road_work.jpg) 
![img4](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Traffic%20Sign%20Classifier/new_images/left_turn.jpeg) ![img5](https://github.com/tranlyvu/autonomous-vehicle-projects/blob/master/Traffic%20Sign%20Classifier/new_images/60_kmh.jpg)

The main difficulty was o resize the image to 32x32x1 to fit into my lenet-based model.

#### Model's predictions on new traffic signs 

Here are the results of the prediction:

| Image			        	  |     Prediction	             | 
|:---------------------------:|:-----------------------------:| 
| Stop Sign      			  | No vehicle					 | 
| Yield sign  				  | Yield sign 					 |
| Road work sign	          | General caution							     |
| Left turn sign      		  | Keep right					 			 |
| Speed limit (26km/h)		  | No passing for vehicles over 3.5 metric tons |


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This does not correspond to the accuracy on the test set.

#### Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

For the first image, the model's first choice was no vehicle sign (0.93) while the correct sign was third rank (0.026)

| Prediction			        	  |     Probability	             | 
|:---------------------------:|:-----------------------------:| 
| Road work     			  | 0.93					 | 
| Traffic signals  			  | 0.038 					 |
| Stop sign	                  | 0.026							     |
| Keep right      		      | 0.00165					 			 |
| Bumpy road		          | 0.0013 |


The model predicted correctly the second image - the Yield sign (almost 1)

| Prediction			        	  |     Probability	             | 
|:---------------------------:|:-----------------------------:| 
| Yield Sign     			  | ~1					 | 
| Children crossing           | ~0 					 |
| End of all speed and passing limits | ~0							     |
| Speed limit (100km/h)     | ~0					 			 |
| Priority road             | ~0 |

Other images can be seen from the notebook

Overrall, the current model is uncertain as it does not predict well with new images. I'm still not sure the reason


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

