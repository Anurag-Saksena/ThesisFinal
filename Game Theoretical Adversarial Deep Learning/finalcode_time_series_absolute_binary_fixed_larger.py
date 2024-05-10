import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import warnings
import copy
import random
from keras import layers, models
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

class CNN:
    @staticmethod
    def build(width, height, depth, classes):
        # model = models.Sequential()
        # inputShape = (width, height, depth)
        # model.add(layers.Conv2D(filters=16, kernel_size=(3,3), padding="valid", activation='leaky_relu', input_shape=inputShape))
        # model.add(layers.AvgPool2D(pool_size=(2,2)))
        # model.add(layers.Conv2D(filters=32, kernel_size=(3,3), padding="valid", activation = 'leaky_relu'))
        # model.add(layers.AvgPool2D(pool_size=(2,2)))
        # # model.add(layers.Dropout(0.2))
        # model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid", activation = 'leaky_relu'))
        # model.add(layers.AvgPool2D(pool_size=(2,2)))
        # # model.add(layers.Conv2D(96, (3,3), activation = 'relu'))
        # # model.add(layers.MaxPooling2D(2,2))
        # # model.add(layers.Dropout(0.2))
        # model.add(layers.Flatten())
        # model.add(layers.Dense(32, activation = 'leaky_relu'))
        # model.add(layers.Dense(32, activation = 'leaky_relu'))
        # # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(1, activation = 'sigmoid'))
        # return model
        model = models.Sequential()
        inputShape = (height, width, depth)
        model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=inputShape))
        model.add(layers.MaxPooling2D(2,2))
        model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
        model.add(layers.MaxPooling2D(2,2))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation = 'relu'))
        model.add(layers.Dense(classes, activation = 'softmax'))
        return model

def train_cnn(train_images, train_labels, val_images = np.array([]),val_labels = np.array([]),epochs = 5):
    model = CNN.build(width=120, height=400, depth=1, classes=2)
    model.summary()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) #98.97 - adam 99.23 - rmsprop
    if val_images.size!=0 and val_labels.size!=0:
        model.fit(train_images, train_labels, epochs = epochs, batch_size = 32,validation_data=(val_images, val_labels))
    else:
        model.fit(train_images, train_labels, epochs = epochs, batch_size = 32)
    return model


def evalute_cnn(model, test_images, test_labels):
    test_loss, test_acc  = model.evaluate(test_images, test_labels)
    print(f'[INFO] Loss:{test_loss:.4f} Test accuracy: {(test_acc*100):.4f}')


def calculate_f1_score(model, test_images,test_labels):
    test_prob = model.predict(test_images)
    test_pred = test_prob.argmax(axis=-1)
    f1 = f1_score(test_labels, test_pred, average='weighted')
    print(f'[INFO] F1-Score is:{f1:.4f}')
    return f1


def get_numpy_array_for_image(image_path):
    # print(image_path)
    img = cv2.imread(image_path)

    img_transformed = img[:, :, :1]

    img_transformed = 255 - img_transformed

    img_final = np.squeeze(img_transformed, axis=2)

    # img_final[0] = 0
    # img_final[-1] = 0

    return img_final


import pandas as pd

def csv_to_array(file_path):
    df = pd.read_csv(file_path)
    return np.array(list(df['data_labels']))


import os

current_directory = os.getcwd()

os.chdir("D:/Pycharm Projects")

train_folder = rf"ThesisFinalData\larger_scaled_images_rolling_window_6\train"

train_images = []

train_images_list = os.listdir(fr'{train_folder}\images')
train_images_list.sort(key=lambda x: int(x.split('_')[1]))

for index, img in enumerate(train_images_list):
    print(index)
    train_images.append(get_numpy_array_for_image(rf'{train_folder}\images\{img}'))

train_images = np.array(train_images)

train_labels = csv_to_array(rf'{train_folder}\train_data_labels_absolute_binary.csv')

test_folder = rf"ThesisFinalData\larger_scaled_images_rolling_window_6\test"

test_images = []

test_images_list = os.listdir(fr'{test_folder}\images')
test_images_list.sort(key=lambda x: int(x.split('_')[1]))

for index, img in enumerate(test_images_list):
    print(index)
    test_images.append(get_numpy_array_for_image(rf'{test_folder}\images\{img}'))

test_images = np.array(test_images)

test_labels = csv_to_array(rf'{test_folder}\test_data_labels_absolute_binary.csv')

# os.chdir(current_directory)


# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 400, 120, 1)
test_images = test_images.reshape(test_images.shape[0], 400, 120, 1)

train_images = train_images.astype('float16')
test_images = test_images.astype('float16')

train_images , test_images = train_images / 255 , test_images / 255
# train_images, test_images = train_images - 0.5, test_images - 0.5


# model = train_cnn(train_images_dummy,train_labels_dummy,test_images_dummy,test_labels_dummy, epochs = 5)
# print(model.predict(test_images_dummy))
# evalute_cnn(model,test_images_dummy,test_labels_dummy)
# calculate_f1_score(model,test_images_dummy,test_labels_dummy)
model = train_cnn(train_images,train_labels,test_images,test_labels, epochs = 5)
print(model.predict(test_images))
# model = train_cnn(train_images,train_labels)
evalute_cnn(model,test_images,test_labels)
calculate_f1_score(model,test_images,test_labels)