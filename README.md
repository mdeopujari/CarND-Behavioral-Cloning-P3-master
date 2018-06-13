# **Behavioral Cloning** 

## Cloning Driving Behavior in an Agent

---
In this project, I am applying what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I am training, validating and testing a model using KERAS. The model is trained using image data and steering angles from a car-driving simulator. After training, the model outputs a steering angle which is then used in the simulator to drive an autonomous vehicle around a track.

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_lane_driving.jpg "Center Lane Driving"
[image3]: ./examples/Left_recovery1.jpg "Recovery Image1"
[image4]: ./examples/Left_recovery2.jpg "Recovery Image2"
[image5]: ./examples/Left_recovery3.jpg "Recovery Image3"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. 

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on traditional LeNet5 architecture. It consists of a convolution neural network with 5x5 filter sizes and depths between 6 and 16 (model.py lines 68-71) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code lines 68-71). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 75). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 86). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving both ways on same track and also focused driving in trouble areas (dirt patch, bridge etc.) 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the LeNet5 model. I thought this model might be appropriate because it is flexible and works with almost any type of image size.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in a 70:30 ratio. I chose this ratio to prevent overfitting. To combat the overfitting, I also added dropout layer in the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like after crossing the bridge where lane markings end. To improve the driving behavior in these cases, I recorded more training data in these areas and drove very slowly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 64-80) consisted of a convolution neural network with the following layers and layer sizes: 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image  							|
| Lambda                | Image Normalization                           |
| Cropping     	        | Cropped top 70 and bottom 25 rows of pixels outputs 65x320x3 	| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 6 feature maps 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride  				                    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 16 feature maps	|
| RELU					|												|
| Max pooling	      	| 2x2 stride 				                    |
| Flatten Layer			|										        |
| Dropout		        | To prevent overfitting 						|
| Fully connected		| outputs 120       							|
| Fully connected		| outputs 84       								|
| Fully connected		| outputs steering angle 						|        		

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to center whenever it nears the edge of the road. These images show what a recovery looks like starting from left edge to the middle :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

After the collection process, I had ~11000 number of data points. I then preprocessed this data by normalizing and cropping out irrelevant data out of image. I did this in network so that we get the same result during inference stage.

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the loss charts converging. I used an adam optimizer so that manually training the learning rate wasn't necessary.
