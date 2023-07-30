# Object Detection with YOLOv3 for Car-Mounted Cameras

![alt text](https://github.com/TejasKalsait/Object_Detection/blob/main/yolo_example.jpg?raw=true)

This is an object detection project using the YOLOv3 algorithm to identify objects of 80 classes through a camera mounted on a car. The project is implemented in a Jupyter Notebook, making it easy to run and experiment with.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)

## Introduction

Object detection is a computer vision task that involves identifying and locating objects of interest within an image or video. YOLO (You Only Look Once) is a popular real-time object detection algorithm that can detect multiple objects in an image simultaneously.

This project focuses on using YOLOv3, which is an improved version of YOLO, to perform real-time object detection using a camera mounted on a car. The algorithm is capable of detecting a wide range of objects belonging to 80 different classes such as pedestrians, cars, traffic signs, bicycles, and more.

## Setup

To run this project, you'll need to set up the following environment:

1. Python >= 3.8
2. Jupyter Notebook
3. Required libraries: OpenCV, NumPy, TensorFlow, etc.

You can install the necessary libraries using `pip`:

```bash
pip install opencv-python numpy tensorflow
```

Next, you'll need to download the YOLOv3 pre-trained weights and configuration files. You can find these files from the official YOLO website (https://pjreddie.com/darknet/yolo/). Place the downloaded files in the appropriate directories.

## Usage

1. Clone this repository to your local machine.
2. Open the Jupyter notebook `object_detection_yolov3.ipynb`.
3. Follow the instructions provided in the notebook to execute each code cell.

The Jupyter notebook contains detailed explanations for each step of the object detection process, making it easy to understand the workflow.

## Dataset

This project uses a dataset that includes images and corresponding annotation files (e.g., XML, COCO, YOLO format). The dataset consists of images captured by a camera mounted on a car, and each image is annotated with bounding boxes and class labels for the objects present in the scene.

Due to the size and licensing restrictions of some datasets, we cannot include the dataset in this repository. However, you can use your own dataset or obtain publicly available datasets suitable for object detection tasks.

## Model

The YOLOv3 model used in this project is pre-trained on a large dataset and can detect objects from 80 different classes with impressive accuracy. The model architecture is optimized for real-time inference, making it suitable for applications such as object detection in driving scenarios.

In the Jupyter notebook, we'll load the pre-trained model and fine-tune it on our specific dataset to adapt it for car-mounted camera object detection.

## Results

The results of the object detection process will be shown in the Jupyter notebook itself. You will see the detected objects in the test images along with their corresponding class labels and confidence scores.

## Thank you
