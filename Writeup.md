# **Traffic Sign Recognition** 

## Writeup

This is my writeup for the project "Traffic Sign Classifier" of Self Driving Car Nanadegree on Udacity.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
## Project code

Here is my working repository for this project:

https://github.com/pl80tech/CarND-Traffic-Sign-Classifier

It is imported from below original repository:

https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project

[//]: # (Image References)

[image1a]: ./output_images/train_data.jpg "Training data"
[image1b]: ./output_images/valid_data.jpg "Validation data"
[image1c]: ./output_images/test_data.jpg "Test data"
[image_m1]: ./output_images/preprocessed_method_1.jpg "Method 1"
[image_m2]: ./output_images/preprocessed_method_2.jpg "Method 2"
[image_m3]: ./output_images/preprocessed_method_3.jpg "Method 3"
[image_m4]: ./output_images/preprocessed_method_4.jpg "Method 4"
[image_m5]: ./output_images/preprocessed_method_5.jpg "Method 5"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This file is my writeup file for this project.

Here is a link to my [project code](https://github.com/pl80tech/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is *34799*
* The size of the validation set is *4410*
* The size of test set is *12630*
* The shape of a traffic sign image is *(32, 32, 3)*
* The number of unique classes/labels in the data set is *43*

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of examples/samples per each class/label on each dataset (training, validation, test).

* Training data (*Red*)
<img src="output_images/train_data.jpg" title="Training data"/>

* Validation data (*Green*)
<img src="output_images/valid_data.jpg" title="Validation data"/>

* Test data (*Blue*)
<img src="output_images/test_data.jpg" title="Test data"/>

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I implemented below preprocessing methods in preprocess() for each image & preprocess_pipeline() for the whole dataset.

* *Method 1*: keep original (not process anything)
* *Method 2*: normalize to change pixel value to be within (-1, 1)
* *Method 3*: change to grayscale then adjust the size (32x32x3) to fit in the model
* *Method 4*: use Gaussian Blur (kernel size = 3) to smooth the image
* *Method 5*: use Gaussian Blur (kernel size = 3) to smooth the image
* *Method 6*: combine the images preprocessed by method 4 and method 1 (to double the number of training data to improve underfitting).

As an example, here are the preprocessed images of a training image with index #500 (method 1 ~ method 5, respectively):

<img src="output_images/preprocessed_method_1.jpg" width=150 title="Method 1"/>
<img src="output_images/preprocessed_method_2.jpg" width=150 title="Method 2"/>
<img src="output_images/preprocessed_method_3.jpg" width=150 title="Method 3"/>
<img src="output_images/preprocessed_method_4.jpg" width=150 title="Method 4"/>
<img src="output_images/preprocessed_method_5.jpg" width=150 title="Method 5"/>

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I implemented below 3 neural network architectures based on LeNet model:

* LeNet_1: simple model - using the filter with size (5, 5, 3, 6) to handle 3-channel input image (32x32x3)
* LeNet_2: broader model - using the filter with bigger size (5, 5, 3, 12) to handle more feature
* LeNet_3: broader & deeper model - using the filter with bigger size (5, 5, 3, 12) and adding one more fully connected layer

After tuning with many parameters, I selected below model (LeNet_3) as final solution. It includes the following layers:

| Layer         	 | Input size | Output size | Description/Note 			 |
|:------------------:|:----------:|:-----------:|:--------------------------:|
| Input         	 | 32x32x3    | -           | RGB image      			 |
| Convolution #1     | 32x32x3 	  | 28x28x12    | 1x1 stride, valid padding  |
| Activation		 | 28x28x12	  | 28x28x12    | Relu  					 |
| Pooling	    	 | 28x28x12	  | 14x14x12    | Max pooling			 |
| Convolution #2     | 14x14x12   | 10x10x25    | 1x1 stride, valid padding  |
| Activation		 | 10x10x25	  | 10x10x25    | Relu  					 |
| Pooling	    	 | 10x10x25	  | 5x5x25	    | Max pooling				 |
| Flatten	    	 | 5x5x25	  | 625  	    | 							 |
| Fully connected #1 | 625		  |	400		    |      						 |
| Activation		 | 400		  | 400		    | Relu  					 |
| Fully connected #2 | 400		  |	240		    |      						 |
| Activation		 | 240		  | 240		    | Relu  					 |
| Fully connected #3 | 240		  |	100		    |      						 |
| Activation		 | 100		  | 100		    | Relu  					 |
| Fully connected #4 | 100		  |	43		    | Output (logits)			 |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Here are the images that show tuning result with different model & hyperparameters:

* Training with different preprocessing method

<img src="output_images/valid_acc_with_different_preprocess.jpg" title="Training result"/>

* Training with different learning rate

<img src="output_images/valid_acc_with_different_rate.jpg" title="Training result"/>

* Training with different batch size

<img src="output_images/valid_acc_with_different_batchsize.jpg" title="Training result"/>

* Training with different model (batch size: 64 & 32)

<img src="output_images/valid_acc_with_different_model.jpg" title="Training result"/>

<img src="output_images/valid_acc_with_different_model_b32.jpg" title="Training result"/>

* Training with different model using more data (batch size: 64 & 32)

<img src="output_images/valid_acc_with_different_model_b32_moredata.jpg" title="Training result"/>

<img src="output_images/valid_acc_with_different_model_b64_moredata.jpg" title="Training result"/>

<img src="output_images/valid_acc_with_different_model_moredata_pre14.jpg" title="Training result"/>

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. I downloaded the whole dataset from following link [GTSRB_Online-Test-Images.zip](http://benchmark.ini.rub.de/Dataset/GTSRB_Online-Test-Images.zip) (Test dataset for online-competition only) and picked the first 5 images which have size 32x32x3 (to fit in my model architecture).

<img src="test_images/00001.jpg" width=150 title="00001.jpg (converted from ppm file)"/>
<img src="test_images/00013.jpg" width=150 title="00013.jpg (converted from ppm file)"/>
<img src="test_images/00090.jpg" width=150 title="00090.jpg (converted from ppm file)"/>
<img src="test_images/00115.jpg" width=150 title="00115.jpg (converted from ppm file)"/>
<img src="test_images/00184.jpg" width=150 title="00184.jpg (converted from ppm file)"/>

The fourth image might be difficult to classify because it has low quality and even difficult for human eye. Other images are clearer so they may be classified correctly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        	|   Result   |
|:---------------------:|:-----------------------------:|:----------:|
| Speed limit (60km/h)  | Speed limit (60km/h)			|Correct     |
| Wild animals crossing | Wild animals crossing 		|Correct     |
| No passing			| No passing					|Correct     |
| Speed limit (120km/h)	| Speed limit (80km/h)			|Wrong       |
| Turn right ahead		| Turn right ahead     			|Correct     |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 89.6%. The predicted label is "8" while the correct label is "6". As shown in the histogram, the number of training data for label "6" is fewer than for other labels. It may be one of the potential point for further improvement.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


