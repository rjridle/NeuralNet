import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle


CWD = os.getcwd()
TRAINING_DIR = CWD + "/dogs-vs-cats/train"
TESTING_DIR = CWD + "/dogs-vs-cats/test1"
IMG_SIZE = 100
mnist = tf.keras.datasets.mnist


def main():

    X, y = mnsit.load_data()

    #model = Sequential()
    #model.add(Conv2D(64, (3,3), input_shape= X.shape[1:]))
    #model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(2,2)))

    #model.add(Conv2D(64, (3,3)))
    #model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(2,2)))

    #model.add(Flatten())
    #model.add(Dense(64))

    #model.add(Dense(1))
    #model.add(Activation('sigmoid'))

    #model.compile(loss="binary_crossentropy",
    #              optimizer="adam",
    #              metrics=['accuracy'])

    #model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)

def CreateTrainingData():
    training_data = []
    X = []
    y = []

    for img in os.listdir(TRAINING_DIR):
        try:
            category = img.split('.')[0]
            img_array = cv2.imread(os.path.join(TRAINING_DIR, img), cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([resized_array, int(category == "cat")])
        except Exception as e:
            print("Broken image. Passing")
            pass

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X/255.0
    y = np.array(y)
    return X, y

def CreateTestingData():
    testing_data = []
    X = []

    for img in os.listdir(TESTING_DIR):
        try:
            category = img.split('.')[0]
            img_array = cv2.imread(os.path.join(TESTING_DIR, img), cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testing_data.append([resized_array, int(category == "cat")])
        except Exception as e:
            print("Broken image. Passing")
            pass

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X/255.0
    y = np.array(y)
    return X, y

def PickleData(X, y):
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

def CreateModel(X, y):

    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape= X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

    model.fit(X, y, batch_size=128, epochs = 10, validation_split=0.1)

    return model


if __name__ == "__main__":
    main()
    
