# Project Title:
End-to-End License Plate Detection and Recognition

## Objective:
The aim of this project is to detect a License Plate(LP) in an image consisting of vehicle and then recognise the corresponding LP number simultaneously.

## Dataset:
In this project we use CCPD, a large and comprehensive LP datasets. The dataset is available here : https://drive.google.com/file/d/1fFqCXjhk7vE9yLklpJurEwP9vdLZmrJd/view

## Approach:
The paper we implemented : http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.pdf

We use 10 Convolutional Neural Network(CNN) layers to extract features from the input image and then use fully connected layers for prediction of the bounding box around the License Plate.The bounding box is parameterized by 4 numbers denoting the center and dimensions(both x and y) of the Bounding Box.For training, we have used the L1 loss between the predicted box and the ground truth box.

After detection of LP we use the "Recognition Module" which exploits Region of Interest(ROI) to extract features map of interest and several classifiers to predict the corresponding license plate number.We have used Cross-Entropy Loss as the Classification Loss.The entire module is a single unified network for License Plate detection and recognition.

The network architecture of our model as follows:
<p align='center'>
  <img src='./Image/model.png' alt='x net'/>
</p>


### Detection Accuracy Metric:
For Detection Accuracy metric the bounding box is considered to be correct if and only if its Intersection-over-Union(IoU) with the ground-truth bounding box is more than 70%(IoU>0.7)

### Recognition Accuracy Metric:
For Recognition Accuracy metric, a License Plate(LP) recognition is considered to be correct if and only if all of the characters in the LP numbers are correctly recognised.

## Results:

#### Detection:
The model’s validation accuracy,on about 20,000 images,(for IoU >= 0.70)= 65.5%.<br>
The model’s validation accuracy,on about 20,000 images,(for IoU >= 0.65)= 80.7%.<br>
The model’s validation accuracy,on about 20,000 images,(for IoU >= 0.60)= 89.6%.<br>
<p align='center'>
  <img src='./Image/0225-88_79-142&561_424&640-413&626_166&636_152&574_398&564-0_0_2_29_10_25_26-111-55.jpg' height="300">
</p>

<p align='center'>
  <img src='./Image/0294875478927-109_64-206&453_411&593-395&588_215&525_207&457_388&520-0_0_24_30_10_32_33-167-62.jpg' height="300" >
</p>
<br>

#### Recognition:
The model's training accuracy = 20.57%<br>
The model's test accuracy = 20.13%

## Instructions:
To start training,run python3 training.py.<br>
Link to our trained model: https://drive.google.com/open?id=1ytZTnKpe2TkN73gjKQjovz7MGTi0kmH-

## Acknowledgements:
This paper: http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.pdf
