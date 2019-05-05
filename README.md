# Project Title:
End-to-End License Plate Detection and Recognition

## Objective:
The aim of this project is to detect a License Plate(LP) in an image consisting of vehicle and then recoginize the corresponding LP number simultaneously.

## Dataset:
In this project we use CCPD, a large and comprehensive LP datasets. The dataset is available here : https://drive.google.com/file/d/1fFqCXjhk7vE9yLklpJurEwP9vdLZmrJd/view

## Approach:
The paper we implemented : http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.pdf

We use 10 Convolutional Neural Network(CNN) layers to extract features from the input image and then use fully connected layers for prediction of the bounding box around the License Plate. After detection of LP we use the "Recognition Module" which exploits Region of Interest(ROI) to extract features map of interest and several classifiers to predict the corresponding license plate number. The entire module is a single unified network for License Plate detection and recognition.

The network architecture of our model as follows:
<p align='center'>
  <img src='./Image/model.png' alt='x net'/>
</p>


### Detection Accuracy Metric:
For Detection Accuracy metric the bounding box is considered to be correct if and only if its Intersection-over-Union(IoU) with the ground-truth bounding box is more than 70%(IoU>0.7)

### Recognition Accuracy Metric:
For Recognition Accuracy metric, a License Plate(LP) recognition is considered to be correct if and only if all of the characters in the LP numbers are correctly recognised.
