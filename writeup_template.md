# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* utils.py: some general functions used in both model.py and drive.py
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py my_model_weights.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

I did not see the need to use dropout as a normalization technique. However, I did use BatchNormalization after each convolution layers, which does introduce a small factor of regularization.



#### 3. Model parameter tuning

The model used an adam optimizer, with a learning rate of 0.0005. (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple CNN architecture then work from this. 

My first step was to use a convolution neural network model similar to the AlexNet. I thought this model might be appropriate because it is widely used, and has been successful for several tasks. From this I introduced several more layers in the network. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model with batch normalization layers. This has a documented effect to combat overfitting, and improve the convergence of the model. I also introduced more data to the network. I recorded data from the initial track, both driving the usual direction, and changing direction. 

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes: 

| Layer                     | Filters | Output size |
| ------------------------- | ------- | ----------- |
| Cropping2D                | --      | 90, 320, 3  |
| Conv2D, ReLU              | 32      | 90, 320, 16 |
| MaxPool2D                 | --      | 45, 160, 16 |
| Conv2D, ReLU              | 32      | 45, 160, 32 |
| Conv2D, ReLU              | 32      | 45, 160, 32 |
| MaxPool2D                 | --      | 22, 80, 32  |
| Conv2D, ReLU              | 64      | 22, 80, 64  |
| Conv2D, ReLU              | 64      | 22, 80, 64  |
| MaxPool2D                 | --      | 11, 40, 64  |
| Conv2D, ReLU              | 128     | 11, 40, 128 |
| Conv2D, ReLU              | 128     | 11, 40, 128 |
| MaxPool2D                 | --      | 5, 20, 128  |
| Conv2D, ReLU              | 256     | 5, 20, 256  |
| MaxPool2D, Stride = [1,2] | --      | 4, 10, 256  |
| Conv2D, ReLU              | 256     | 4, 10, 256  |
| MaxPool2D                 | --      | 2, 5, 256   |
| Conv2D, ReLU              | 512     | 2, 5, 512   |
| MaxPool2D                 | --      | 1, 2, 512   |
| Flatten                   | --      | 1024        |
| Dense, ReLU               | 32      | 32          |
| Dense                     | 1       | 1           |

Each Conv2D layer is followed by a ReLU activation and BatchNorm2d. 

The second to last Dense layer is used with a ReLU activation. 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
