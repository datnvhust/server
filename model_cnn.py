import tensorflow as tf
import numpy as np
import pickle
from tensorflow import keras
from keras_radam.training import RAdamOptimizer
import build_data_label
import test
import testttt
import pandas as pd
from focal_loss import BinaryFocalLoss

# define file name
BUG_MATRIX = 'data/Bug_matrix_AspectJ.txt'
SOURCE_MATRIX = 'data/matrix_sourceAspectj.pickle'

SENT_BUG = 114  # 36  # 110# 114#
SENT_SOURCE = 194  # 134  # 201# 194#


def split_data(matrixs, size):
    bugs = []
    sources = []
    for matrix in matrixs:
        bugs.append(matrix[:size][:])
        sources.append(matrix[size:][:])
    return np.asarray(bugs), np.asarray(sources)


class multipleModel(tf.keras.Model):
    def __init__(self):
        super(multipleModel, self).__init__()

        # for input 1 -- bugs
        self.inp1 = keras.Input(shape=(SENT_BUG, 300, 1,))
        self.conv1 = keras.layers.Conv2D(100, (2, 300), activation='tanh', input_shape=(SENT_BUG, 300, 1,))
        self.pool1 = keras.layers.MaxPool2D((SENT_BUG - 1, 1))

        self.conv2 = keras.layers.Conv2D(100, (3, 300), activation='tanh', input_shape=(SENT_BUG, 300, 1,))
        self.pool2 = keras.layers.MaxPool2D((SENT_BUG - 2, 1))

        self.conv3 = keras.layers.Conv2D(100, (4, 300), activation='tanh', input_shape=(SENT_BUG, 300, 1,))
        self.pool3 = keras.layers.MaxPool2D((SENT_BUG - 3, 1))

        # for input 2 -- sources
        self.inp2 = keras.Input(shape=(SENT_SOURCE, 300, 1,))
        self.conv4 = keras.layers.Conv2D(100, (2, 300), activation='tanh', input_shape=(SENT_SOURCE, 300, 1,))
        self.pool4 = keras.layers.MaxPool2D((SENT_SOURCE - 1, 1))

        self.conv5 = keras.layers.Conv2D(100, (3, 300), activation='tanh', input_shape=(SENT_SOURCE, 300, 1,))
        self.pool5 = keras.layers.MaxPool2D((SENT_SOURCE - 2, 1))

        self.conv6 = keras.layers.Conv2D(100, (4, 300), activation='tanh', input_shape=(SENT_SOURCE, 300, 1,))
        self.pool6 = keras.layers.MaxPool2D((SENT_SOURCE - 3, 1))

        # general
        self.flatten = keras.layers.Flatten()
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
        x = self.pool1(x)
        # out1 = self.reshape(x)
        out1 = self.flatten(x)

        x = self.conv2(x1)
        x = self.pool2(x)
        # out2 = self.reshape(x)
        out2 = self.flatten(x)

        x = self.conv3(x1)
        x = self.pool3(x)
        # out3 = self.reshape(x)
        out3 = self.flatten(x)

        concatenate = self.concatenate([out1, out2])
        output1 = self.concatenate([concatenate, out3])

        # input 2 ===============================================================
        x = self.conv4(x2)
        x = self.pool4(x)
        # out1 = self.reshape(x)
        out1 = self.flatten(x)

        x = self.conv5(x2)
        x = self.pool5(x)
        # out2 = self.reshape(x)
        out2 = self.flatten(x)

        x = self.conv6(x2)
        x = self.pool6(x)
        # out3 = self.reshape(x)
        out3 = self.flatten(x)

        concatenate = self.concatenate([out1, out2])
        output2 = self.concatenate([concatenate, out3])

        # multiple inputs --> 1 output
        concatenate = self.concatenate([output1, output2])
        x = self.flatten(concatenate)
        # x = self.dense1(x)
        # x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense4(x)
        return x


class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        x1, x2 = x

        with tf.GradientTape() as tape:
            y_pred = self((x1, x2), training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            print("loss: ", loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def model_run():
    # define loss function, optimize, model
    lossFunction = tf.keras.losses.CategoricalCrossentropy()  # custom_loss_optimization_function.CategoricalCE(w)
    opt = RAdamOptimizer(learning_rate=1e-4)

    model_CNN = multipleModel()  # CustomModel(input1, input2, output)
    model_CNN.compile(optimizer=opt,
                      loss=lossFunction,
                      metrics=["acc"])

    # fit data in model ===========================================================================================
    # because AspectJ is not too big, split bug report into 3 folds, fold 1 is oldest, for 3 is newest.
    # Else, projects will split 10 folds
    # fold 1 is split into 60% training, 40% validation. And fold 2 training - fold 3 test
    # (fold 3 training - fold 4 test, fold 4 training - fold 5 test, ....)

    file = open(BUG_MATRIX, 'rb')
    bug_matrix = pickle.load(file)

    # # get the minimize bugs
    # bug_matrix = bug_matrix[:50]

    file.close()

    file = open(SOURCE_MATRIX, 'rb')
    source_matrix = pickle.load(file)
    file.close()

    num_fold = 3
    distance = len(bug_matrix) // num_fold

    # fold 1 is split into 60% training, 40% validation ==========================================================
    fold_1 = bug_matrix[:distance]
    train = distance * 6 // 10
    bug_train = fold_1[:train]
    bug_val = fold_1[train:]

    # get data train and validation
    matrix_train, label_train = build_data_label.get_matrix_and_label(0, train)
    matrix_val, label_val = build_data_label.get_matrix_and_label(train, distance)

    bug_train, source_train = split_data(matrix_train, SENT_BUG)  # 114 is the height of bug matrix
    bug_val, source_val = split_data(matrix_val, SENT_BUG)

    # convert data to array
    bug_train = np.reshape(bug_train, (-1, SENT_BUG, 300, 1))
    bug_val = np.reshape(bug_val, (-1, SENT_BUG, 300, 1))
    source_train = np.reshape(source_train, (-1, SENT_SOURCE, 300, 1))
    source_val = np.reshape(source_val, (-1, SENT_SOURCE, 300, 1))
    bug_train = np.array(bug_train)
    bug_val = np.array(bug_val)
    source_train = np.array(source_train)
    source_val = np.array(source_val)
    label_train = np.array(label_train)
    label_val = np.array(label_val)

    model_CNN.fit(x=(bug_train, source_train), y=label_train, epochs=20, batch_size=32,
                  validation_data=([bug_val, source_val], label_val))

    # fold 2 --> fold 10: fold k is training data, fold k+1 is test data, k = 2..9 ==============================
    for fold in range(2, num_fold, 1):
        train_l = (fold - 1) * distance
        train_r = fold * distance

        test_l = fold * distance
        test_r = min((fold + 1) * distance, len(bug_matrix) - 1)  # out of array

        # get data train and validation
        matrix_train, label_train = build_data_label.get_matrix_and_label(train_l, train_r)
        matrix_test, label_test = build_data_label.get_matrix_and_label_test(test_l, test_r)

        bug_train, source_train = split_data(matrix_train, SENT_BUG)  # 114 is the height of bug matrix
        bug_test, source_test = split_data(matrix_test, SENT_BUG)

        # convert data to array
        bug_train = np.reshape(bug_train, (-1, SENT_BUG, 300, 1))
        bug_test = np.reshape(bug_test, (-1, SENT_BUG, 300, 1))
        source_train = np.reshape(source_train, (-1, SENT_SOURCE, 300, 1))
        source_test = np.reshape(source_test, (-1, SENT_SOURCE, 300, 1))
        bug_train = np.array(bug_train)
        bug_test = np.array(bug_test)
        source_train = np.array(source_train)
        source_test = np.array(source_test)
        label_train = np.array(label_train)
        label_test = np.array(label_test)

        model_CNN.fit(x=(bug_train, source_train), y=label_train, epochs=20, batch_size=64)

        # test model for each fold
        (loss, acc) = model_CNN.evaluate([bug_test, source_test], label_test)
        print("==========>>>> [INFO] test accuracy: {:.4f}, loss: {:.4f}".format(acc, loss), end="")

        predict = model_CNN.predict([bug_test, source_test])

        for i in range(10):
            print(label_test[i], '==', predict[i])

        count = 0
        for i in label_test:
            if i[0] == 1:
                count += 1
        count_pre = 0
        for i in predict:
            if i[0] > 0.5:
                count_pre += 1
        print("label 1: ", count, count_pre)

        count = 0
        for i in label_test:
            if i[0] == 0:
                count += 1
        count_pre = 0
        for i in predict:
            if i[0] < 0.5:
                count_pre += 1
        print("label 0: ", count, count_pre)

        test.test(test_l, test_r, model_CNN)
        test.metrics_evaluate(test_l, test_r, model_CNN)
