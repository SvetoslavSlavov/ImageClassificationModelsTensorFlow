# Image Classification For University
This projects works with https://www.kaggle.com/c/dogs-vs-cats/data they have to be 2000 total and 1000 for each dogs and cats folder. They have to be in a main folder data and subfolders(validation/cat, validation/dogs and the same for train). In validation there have to be 400 extra photos.

1.fit_generator for training Keras a model using Python data generators
2.ImageDataGenerator for real-time data augmentation
3.layer freezing and model fine-tuning

ReLU stack of 3 convolution layers 
Yann LeCun advocated in the 1990s for image classification
binary_crossentropy loss 
to train our model. 

The method use does not repeat photos(shuffle: false) and create
there copies which it rotates. The method use pretrained database (VGG16 network - ImageNet dataset). 

The folder structure 
date/train/ for training 
data/validation/ for testing the algorithm

epochs - for one iteration images
batch_size - all iterations
