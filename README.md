# Object-Classification-and-Localization-with-TensorFlow
This is a multiclass image classification & localization project for SINGLE object using CNN's and TensorFlow API (no Keras) on Python3.

# Steps

1 ) Collecting images via [Google Image Download](https://github.com/hardikvasa/google-images-download). Only one object must be in the image.

<p align="center">
<img src = "https://github.com/MuhammedBuyukkinaci/Object-Classification-and-Localization-with-TensorFlow/blob/master/repository_images/Screenshot%20(36).png" width="800" height="400">
</p>

2 ) Labeling images via [LabelImg](https://github.com/hardikvasa/google-images-download)

<p align="center">
<img src = "https://github.com/MuhammedBuyukkinaci/Object-Classification-and-Localization-with-TensorFlow/blob/master/repository_images/Screenshot%20(35).png" width="800" height="400">
</p>

3 ) Data Augmentation (**create_training_data.py**). Mirroring with respect to x axis, mirroring with respect to y axis and adding noise were carried out. Hereby, data amount were 8-fold.

4 ) After data augmentation, **create_training_data.py** script is creating suitable xml files for augmented images(in order not to label all augmented labels).

5 ) Making our data tabular. Input is image that we feed into CNN. Output1 is one hot encoded classification output. Output2 is the locations of bounding boxes(regression) in **create_training_data.py**.

6 ) Determining hypermaraters in **train.py**.

<p align="center">
<img src = "https://github.com/MuhammedBuyukkinaci/Object-Classification-and-Localization-with-TensorFlow/blob/master/repository_images/hyperparameters.png" 
</p>

7 ) Separating labelled data as train and CV in **train.py**.

<p align="center">
<img src = "https://github.com/MuhammedBuyukkinaci/Object-Classification-and-Localization-with-TensorFlow/blob/master/repository_images/train-cv.png" 
</p>

8 ) Defining our architecture in **train.py**. I used [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) for model architecture.

9 ) Creating 2 heads for calculating loss in **train.py**. One head is classification loss. The other head is regression loss.

<p align="center">
<img src = "https://github.com/MuhammedBuyukkinaci/Object-Classification-and-Localization-with-TensorFlow/blob/master/repository_images/losses.png" 
</p>

10 ) Training the CNN on a GPU (GTX 1050 - One epoch lasted 10 seconds approximately)

11 ) Testing on unseen data (**testing_images** folder) colled from the Internet(in **test.py**).

# Architecture

AlexNet is used as architecture. 5 convolution layers and 3 Fully Connected Layers with 0.5 Dropout Ratio. 60 million Parameters.
![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/alexnet_architecture.png)
