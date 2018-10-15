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
* *Method 6*: combine the images preprocessed by method 4 and method 1 (to double the number of training data to improve the performance).

As an example, here are the preprocessed images of a training image with index #500 (method 1 ~ method 5, respectively):

<img src="output_images/preprocessed_method_1.jpg" width=150 title="Method 1"/> <img src="output_images/preprocessed_method_2.jpg" width=150 title="Method 2"/> <img src="output_images/preprocessed_method_3.jpg" width=150 title="Method 3"/> <img src="output_images/preprocessed_method_4.jpg" width=150 title="Method 4"/> <img src="output_images/preprocessed_method_5.jpg" width=150 title="Method 5"/>

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

I created the function pipeline() to train the defined models with multiple settings & hyperparameters like preprocessing method (#1 ~ #6), learning rate, batch size, number of epochs. I used Adam optimizer as default optimizer. I plotted the training results in project file showing validation accuracy on each epoch for each combination. 

Here are the images that show the training results with different settings & hyperparameters:

* Training with different learning rate
 
	The result with learning rate = 0.001 was stable and better than other learning rate.
 
<img src="output_images/valid_acc_vs_different_rate.jpg" title="Accuracy vs rate"/>

* Training with different batch size

	There was no much difference when using different batch size (at least when training with the defined settings & parameters) so I chose 32 as default size for further tuning.

<img src="output_images/valid_acc_vs_different_batchsize.jpg" title="Accuracy vs batch size"/>

* Training with different preprocessing method

 	Preprocessing method #6 showed better result than other preprocessing method, even reached the target validation accuracy (0.93) so I chose this method for further tuning.

<img src="output_images/valid_acc_vs_different_preprocess.jpg" title="Accuracy vs preprocessing method"/>

* Training with different number of epoch

 	As far as I observed (also shown in above training result), the validation accuracy doesn't change much after 15 epochs so I used epoch around 20 ~ 30 for further tuning.

* Training with different model

 	Model LeNet_3 showed slightly better result than other models, even reached the target validation accuracy (0.93) so I chose this model as final solution.

<img src="output_images/valid_acc_vs_different_model.jpg" title="Accuracy vs model"/>

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Here are the main approaches I have been trying until finding the solution:

* Starting with the sample LeNet model, I customized the model architecture & created the first model (LeNet_1) that can suppport input image size (32x32x3).
* I trained LeNet_1 with various hypeparameters & preprocessing methods (#1 ~ #5) to find the points for further improvement.
* I created more complicated model (LeNet_2 & LeNet_3) and generated additional training data (preprocessing method #6) to improve the validation accuracy.

After selecting the model, preprocessing method & hyperparameters, I ran again the whole process (in section *"Training with selected parameters after tuning (Final solution)"*) which automatically stopped when validation acccuracy reached target value (0.93). Here are the accuracy of my final solution:

* Training set accuracy: **0.988**
* Validation set accuracy: **0.932**
* Test set accuracy: **0.914**

<img src="output_images/validation_accuracy_final.jpg" title="Validation accuracy of final solution"/>

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. I downloaded the whole dataset from following link [GTSRB_Online-Test-Images.zip](http://benchmark.ini.rub.de/Dataset/GTSRB_Online-Test-Images.zip) (Test dataset for online-competition only) and picked the first 5 images which have size 32x32x3 (to fit in my model architecture).

<img src="test_images/5_German_signs.jpg" title="5 test images (German signs)"/>

Here are the quality of each image and the comparison with some random training images of the same label.

* Image #1 (00001.ppm, label = 3)

	Although there is a part of triangle sign in the background, the main sign (60km) inside the image is not much different in terms of direction, brightness, contrast when comparing with below training images. The whole quality is quite good so the model may predict it well.

<img src="output_images/train_images_label_3.jpg" title="Training images with label 3"/>

* Image #2 (00013.ppm, label = 31)

	The background is dark and the sign has small angle so it may be a little bit difficult to predict. However, it still has better quality (in general) when comparing with some of training images below.

<img src="output_images/train_images_label_31.jpg" title="Training images with label 31"/>

* Image #3 (00090.ppm, label = 9)

	The sign is blur but it still has better quality then some training images below. There is a part of triangle sign above the main sign but it looks similar with one of the training image so perhaps the model may learn & classify it well.

<img src="output_images/train_images_label_9.jpg" title="Training images with label 9"/>

* Image #4 (00115.ppm, label = 8)

	It has low quality and even difficult for human eye for prediction. However, some training images also have similar quality. The sign itselfs (Speed limit 120km/h) also looks similar with label = 7 (Speed limit 100km/h) so it may be difficult for the model to classify. The top 5 softmax probabilities also shows the confident level (62% for label 8 & 38% for label 7).

<img src="output_images/train_images_label_8.jpg" title="Training images with label 8"/>

* Image #5 (00184.ppm, label = 33)

	The main sign (inside the image) has quite good quality like brightness, contrast and it looks similar with below training images so the model may be able to classify it well.

<img src="output_images/train_images_label_33.jpg" title="Training images with label 33"/>

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

As shown in project file, here are the results of the prediction:

| Image			        			|     Prediction	        	|   Result   |
|:---------------------------------:|:-----------------------------:|:----------:|
| 3 - Speed limit (60km/h) 			| 3 - Speed limit (60km/h)		|Correct     |
| 31 - Wild animals crossing 		| 31 - Wild animals crossing 	|Correct     |
| 9 - No passing					| 9 - No passing				|Correct     |
| 8 - Speed limit (120km/h)			| 8 - Speed limit (120km/h)		|Correct     |
| 33 - Turn right ahead				| 33 - Turn right ahead     	|Correct     |

Although the confident level for each prediction varied from 62% to nearly 100%, the model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This result makes sense since the accuracy on the test set is 91.4% and the prediction was done on only few samples.

**Note**: I actually couldn't find the correct labels for the test images downloaded from the web so I created the "correct labels" for above comparison based on the judgement by my eye. The 4th image is actually not clear enough to judge so it may affect the prediction result. The top 5 softmax probabilities when predicting 4th image (mentioned in below section) also shows the confident level of the prediction is not high as that of other images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in below sections in [project file](https://github.com/pl80tech/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

* Predict the Sign Type for Each Image
* Analyze Performance
* Output Top 5 Softmax Probabilities For Each Image Found on the Web

Here are the top five softmax probabilities for each image calculated by the model. The result is picked from project file.

```python
Image #1
Top 5 softmax probabilities
[  1.00000000e+00   1.34103232e-19   0.00000000e+00   0.00000000e+00   0.00000000e+00]
Correspondent labels
[3 5 0 1 2]

Image #2
Top 5 softmax probabilities
[  1.00000000e+00   5.41257594e-10   1.28611076e-12   1.55874513e-13   1.25619757e-13]
Correspondent labels
[31 23 15 29 17]

Image #3
Top 5 softmax probabilities
[ 1.  0.  0.  0.  0.]
Correspondent labels
[9 0 1 2 3]

Image #4
Top 5 softmax probabilities
[  6.16894305e-01   3.83105576e-01   3.76517875e-08   4.30486580e-09   2.39555265e-09]
Correspondent labels
[ 8  7  9 16  1]

Image #5
Top 5 softmax probabilities
[ 1.  0.  0.  0.  0.]
Correspondent labels
[33  0  1  2  3]
```

For the 4th image (label 6), the model is only 61.7% confident about its prediction (label 8). The top five soft max probabilities for image #4 are as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.616894     			| 8 - Speed limit (120km/h)						| 
| 0.383106 				| 7 - Speed limit (100km/h)						|
| 3.76518e-08			| 9 - No passing 								|
| 4.30487e-09  			| 16 - Vehicles over 3.5 metric tons prohibited	|
| 2.39555e-09		    | 1 - Speed limit (30km/h)						|

For other images, the max probability is very near to 1.0 which means the model is nearly 100% confident about the predicted result.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


