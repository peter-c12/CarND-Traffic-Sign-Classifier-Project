# **Traffic Sign Recognition** 

## Writeup

Peter Chen

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/training_set_counts.png "Visualization"
[image2]: ./plots/random_examples.png "Visualization"
[image3]: ./plots/grayscale1.png "Grayscaling"
[image4]: ./plots/grayscale2.png "Grayscaling"
[image5]: ./plots/grayscale3.png "Grayscaling"
[image6]: ./images_signs/14.sign1.png "Five Examples"
[image7]: ./images_signs/17.sign2.png "Five Examples"
[image8]: ./images_signs/31.sign3.png "Five Examples"
[image9]: ./images_signs/27.sign4.png "Five Examples"
[image10]: ./images_signs/26.sign5.png "Five Examples"

---
#### Writeup / README

The submitted files includes Traffic_Sign_Classifier.ipynb, images_signs folder which contains the images tested in this projects, and writeup_traffic_sign_classifier.md.

#### Data Set Summary & Exploration

##### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43.

##### 2. Include an exploratory visualization of the dataset.

Here are two exploratory visualizations of the data set. The first visulization is a bar chart showing the sign names and traning samples. The second one shows the 10 random samples of 43 signs.  

![alt text][image1]
![alt text][image2]

#### Design and Test a Model Architecture

##### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because of signal to noise. For many applications of image processing, color information doesn't help us identify important edges or other features. The preprocessing approach normalizes images from [0, 255] to [0, 1], and grayscales is shown in scripts. Here are an example of a traffic sign image before and after normalization and grayscaling.

![alt text][image3]
![alt text][image4]
![alt text][image5]

The train, valid and test data are prepreocessed. Cross validation is used to split training data. To cross validate the model, the given training sets are randomly split into training set and validation set. 10% data for validation is preserved. sklearn has the handy tool train_test_split to do the work.


##### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

LeNet architecture is adapted: Two convolutional layers followed by one flatten layer, drop out layer, and three fully connected linear layers. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
|                       |                                               |
| Input         		| 14x14x12   							        | 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x25 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x25 			    	|
|                       |                                               |
| Flatten               | 5x5x25 => 625									|
| 						|												|
| Dropout               | 625 => 625      								|
|						|			        							|
| Linear				| 625 => 300 									|
|						|												|
| Linear				| 300 => 150 									|
|						|												|
| Linear				| 150 => 43 									|
|						|												|


##### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer , learning rate of 0.001, epochs of 10 and batch size of 64.

##### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
To train the model, I started from a well known architecture (LeNet) because of simplicity of implementation and because it performs well on recognition task with tens of classes (such as charachter recognition as in the practice of LeNet in class). 

* What were some problems with the initial architecture?
After a few runs with this architecture the model tended to overfit to the original training set, in fact the learning curve showed that the training error converged to 99% while the validation error wasn't giving a satisfactory performance.

* How was the architecture adjusted and why was it adjusted? 
Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

* Which parameters were tuned? How were they adjusted and why?
Pooling and Dropout were tuned. Poooling was tested by using max pooling and average pooling; Dropout was tested by using different dropout rates. Tuning these parameters is to avoid overfitting of the model.  

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Choose activation function RELU, pooling, and dropout are important design choices becasue choosing these steps can avoid the overfitting or underfitting of the chosen model.


If a well known architecture was chosen:
* What architecture was chosen?
The final model is shown in Question 2.

* Why did you believe it would be relevant to the traffic sign application?
After tuning the importnat parameters, the accuracies for training, validation, and test sets tend to be consistent. There is no situation happened that one is too high or the others are too low. It is concluded that the model would fit to the traffic sign application. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
My final model results were:
* training set accuracy of 0.995.
* validation set accuracy of 0.942.
* test set accuracy of 0.920.



#### Test a Model on New Images

##### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10]

The first image "Stop Sign" might be difficult to classify because it has the similar frame as speed limit signs.  

##### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Speed limit (30km/h)   						| 
| No entry    			| No entry 										|
| Traffic signals		| Traffic signals								|
| Pedestrians	      	| Pedestrians				 					|
| Wild animals crossing	| Wild animals crossing      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.920; therefore, there are some images on the web that this model will not be able to recognize. Five samples may not be enough to predict new images. More new images such as 20 images can be tested later. In this five images test, it shows the predictions is somewhat underfitted and training set may be overfitted. 

##### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the final cell of the Jupyter notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.983), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .983         			| Stop sign   									| 
| .012     				| No entry  									|
| .004					| Traffic signals								|
| .002	      			| Pedestrians					 				|
| .001				    | Wild animals crossing      					|



#### Reference:
1. https://www.quora.com/In-image-processing-applications-why-do-we-convert-from-RGB-to-Grayscale

