import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle


CWD = os.getcwd()
DATA_DIR = CWD + "/dogs-vs-cats/train"
IMG_SIZE = 100

def main():

    print("start")
    X, y = CreateTrainingData()
    print("data created")
    PickleData(X, y)
    print("data pickled")

    print("1")
    model = Sequential()
    print("2")
    model.add(Conv2D(64, (3,3), input_shape= X.shape[1:]))
    print("3")
    model.add(Activation("relu"))
    print("4")
    model.add(MaxPooling2D(pool_size=(2,2)))
    print("5")

    model.add(Conv2D(64, (3,3)))
    print("6")
    model.add(Activation("relu"))
    print("7")
    model.add(MaxPooling2D(pool_size=(2,2)))
    print("8")

    model.add(Flatten())
    print("9")
    model.add(Dense(64))
    print("10")

    model.add(Dense(1))
    print("11")
    model.add(Activation("sigmoid"))
    print("12")

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    print("13")

    model.fit(X, y, batch_size=32, validation_split=0.1)
    print("14")

def CreateTrainingData():
    training_data = []
    X = []
    y = []

    for img in os.listdir(DATA_DIR):
        try:
            category = img.split('.')[0]
            img_array = cv2.imread(os.path.join(DATA_DIR, img), cv2.IMREAD_GRAYSCALE)
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
    return X, y

def PickleData(X, y):
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

#def CreateModel(X, y):
#
#    model = Sequential()
#    model.add(Conv2D(64, (3,3), input_shape= X.shape[1:]))
#    model.add(Activation("relu"))
#    model.add(MaxPooling2D(pool_size=(2,2)))
#
#    model.add(Conv2D(64, (3,3)))
#    model.add(Activation("relu"))
#    model.add(MaxPooling2D(pool_size=(2,2)))
#
#    model.add(Flatten())
#    model.add(Dense(64))
#
#    model.add(Dense(1))
#    model.add(Activation('sigmoid'))
#
#    model.compile(loss="binary_crossentropy",
#                  optimizer="adam",
#                  metrics=['accuracy'])
#
#    model.fit(X, y, batch_size=32, validation_split=0.1)


if __name__ == "__main__":
    main()
    
