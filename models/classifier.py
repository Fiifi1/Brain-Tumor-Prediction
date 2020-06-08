#'''Importing all needed libraries from tensorflow and keras'''

#Main CNN model can be retrained again on any MRI dataset

import tensorflow as tf                                        
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#Setting up the CNN 

detector = Sequential()

#Convolution is responsible for extracting appropriate features from image
#RELU - Sets all negative pixel values to zero for easier classification
detector.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))

#Pooling: Reduces dimensions of the feature maps
detector.add(MaxPooling2D(pool_size=(2, 2)))

#Second layer, including convulution, pooling and flattening in order
#set 2D volumes to 1D
detector.add(Conv2D(32, (3, 3), activation='relu'))
detector.add(MaxPooling2D(pool_size=(2, 2)))
detector.add(Flatten())

#Fully connected layer
detector.add(Dense(units=128, activation='relu'))
detector.add(Dense(units=2, activation='sigmoid'))

#importing data and setting data path
data_path = './data/train'

#use image generator to rescale image and allow flipping of image
train_data = ImageDataGenerator(rescale=1.0/255.0, shear_range = 0.2,
                                zoom_range = 0.2, horizontal_flip = True)

generator = train_data.flow_from_directory(data_path, 
                                        target_size=(224, 224),
                                        color_mode = 'rgb', 
                                        batch_size= 16,
                                        class_mode = 'categorical', 
                                        shuffle = True)

#set up classifier with optimizer, loss function and classification metrics
detector.compile(optimizer='Adam', loss = 'binary_crossentropy', 
                metrics=['accuracy'])

step_size = generator.n // generator.batch_size

hist = detector.fit_generator(generator = generator, 
                            steps_per_epoch = step_size,
                            epochs = 50)

#Generate Loss graph
plt.plot(hist.history['loss'])
plt.title('Classifier Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.savefig('Loss.jpg')

#Generate Classifier accuracy graph
plt.plot(hist.history['accuracy'])
plt.title('Classifier Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('accuracy.jpg')

#Save classifier weights
detector.save('classifier.h5')