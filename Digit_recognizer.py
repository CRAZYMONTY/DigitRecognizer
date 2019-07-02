from sklearn.datasets import load_digits
import keras 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#### Loading Data #####

data = load_digits()

x = data.data
y = data.target

x = x.reshape((1797,8,8,1))

print("x Shape : ")
print(x.shape)

x_train,x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2)

y_train = keras.utils.to_categorical(y_train, 10)
print("y_train Shape : ")
print(y_train.shape)

#### Designing our CNN architecture ####

classifier = Sequential()     													# Sequential layer with input size (8x8) Image

classifier.add(Conv2D(32,(3,3),activation='relu',input_shape=(8,8,1)))			# Convolutional 2D layer with 32-(3x3) filters
classifier.add(MaxPool2D(pool_size=(2,2)))										# Applying Max pooling with (2x2) kernel
classifier.add(Flatten())

classifier.add(Dense(units=228,activation='relu'))								# Linearly seprable features passed to fully connected layer
classifier.add(Dense(10,activation='softmax'))									# Final layer(output)

# Compilation - Using Adam Optimizer with loss is categorical_crossentropy

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print("Training Model")

classifier.fit(x=x_train,y=y_train,batch_size=10,epochs=10)

#### Evaluating Model ####

print("Classification report")
print(classification_report(y_test,classifier.predict_classes(x_test)))

print("Confusion Matrix")
print(confusion_matrix(y_test,classifier.predict_classes(x_test)))
