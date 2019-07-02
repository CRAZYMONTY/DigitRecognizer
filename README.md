# DigitRecognizer

Python Script with Convolutional Neural Network Digit Recognizer implemented in Keras 🖼️.

This code implements **Convolutional Neural Network(CNN)** to classify/recognize digits in an image of 8x8 dim.

The Architecture of the model is Descirbed below.

______________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 6, 6, 32)          320       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 3, 3, 32)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 288)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 228)               65892     
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2290      
=================================================================
Total params: 68,502
Trainable params: 68,502
Non-trainable params: 0

# Outputs

![alt text](https://github.com/montySaini25/DigitRecognizer/blob/master/outputs/label_0.png)

![alt text](https://github.com/montySaini25/DigitRecognizer/blob/master/outputs/label_4.png)

![alt text](https://github.com/montySaini25/DigitRecognizer/blob/master/outputs/label_5.png)

![alt text](https://github.com/montySaini25/DigitRecognizer/blob/master/outputs/label_6.png)

![alt text](https://github.com/montySaini25/DigitRecognizer/blob/master/outputs/label_9.png)

