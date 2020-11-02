import tensorflow as tf
import numpy as np
import pickle
from tensorflow import keras
from keras_radam.training import RAdamOptimizer

SENT_BUG = 114  # 36  # 110# 114#
SENT_SOURCE = 194  # 134  # 201# 194#


class multipleModel(tf.keras.Model):
    def __init__(self):
        super(multipleModel, self).__init__()

        # for input 1 -- bugs
        self.inp1 = keras.Input(shape=(SENT_BUG, 300, 1,))
        self.conv1 = keras.layers.Conv2D(100, (3, 100), activation='tanh', input_shape=(SENT_BUG, 300, 1,))
        self.conv2 = keras.layers.Conv2D(100, (7, 7), activation='tanh')
        self.conv3 = keras.layers.Conv2D(100, (5, 5), activation='tanh')
        self.conv4 = keras.layers.Conv2D(100, (3, 3), activation='tanh')

        self.pool1 = keras.layers.MaxPool2D((SENT_BUG - 18 + 4, 1))
        self.pool1a = keras.layers.MaxPool2D((SENT_BUG - 18 + 4, 300 - 115 + 4))

        # for input 2 -- sources
        self.inp2 = keras.Input(shape=(SENT_SOURCE, 300, 1,))
        self.conv5 = keras.layers.Conv2D(100, (3, 100), activation='tanh', input_shape=(SENT_SOURCE, 300, 1,))
        self.conv6 = keras.layers.Conv2D(100, (7, 7), activation='tanh')
        self.conv7 = keras.layers.Conv2D(100, (5, 5), activation='tanh')
        self.conv8 = keras.layers.Conv2D(100, (3, 3), activation='tanh')

        self.pool2 = keras.layers.MaxPool2D((SENT_SOURCE - 18 + 4, 1))
        self.pool2a = keras.layers.MaxPool2D((SENT_SOURCE - 18 + 4, 300 - 115 + 4))

        # general
        self.flatten = keras.layers.Flatten()
        self.densea = keras.layers.Dense(100, activation='tanh')
        self.reshape = keras.layers.Reshape((1, 100))
        self.concatenate = keras.layers.Concatenate()
        self.dense1 = keras.layers.Dense(128, activation='sigmoid')
        self.dense2 = keras.layers.Dense(64, activation='sigmoid')
        self.dense3 = keras.layers.Dense(32, activation='sigmoid')
        self.dense4 = keras.layers.Dense(2, activation='softmax')
        self.dropout1 = keras.layers.Dropout(0.3)
        self.dropout2 = keras.layers.Dropout(0.5)

    def call(self, data):
        x1, x2 = data
        # input 1
        x = self.conv1(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool1a(x)
        x = self.flatten(x)
        x = self.densea(x)
        output1 = self.reshape(x)

        # input 2 ===============================================================
        x = self.conv5(x2)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool2a(x)
        x = self.flatten(x)
        x = self.densea(x)
        output2 = self.reshape(x)

        # multiple inputs --> 1 output
        concatenate = self.concatenate([output1, output2])
        x = self.flatten(concatenate)
        # x = self.dense1(x)
        # x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense4(x)
        return x
