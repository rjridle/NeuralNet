import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle


CWD = os.getcwd()
TRAINING_DIR = CWD + "/data/dogs-vs-cats/train"
TESTING_DIR = CWD + "/data/dogs-vs-cats/test1"
PICKLE_DIR = CWD + "/pickle"
MODELS_DIR = CWD + "/models"
IMG_SIZE = 200
ACTIVATION = "relu"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 5
VALIDATION_SPLIT = 0.1


def main():

    X_train, y_train = CreateTrainingData()
    DC_model = CreateModel(X_train, y_train, ACTIVATION, OPTIMIZER, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT)
    DC_model.save(MODELS_DIR + "/DC_INCREASING_IMSZ-" + str(IMG_SIZE) + "_ACT-" + str(ACTIVATION) + "_BSZ-" + str(BATCH_SIZE) + "_EP-" + str(EPOCHS) + "_VALSPLT-" + str(VALIDATION_SPLIT))

    #DC_model = keras.models.load_model(MODELS_DIR + "/DogsVsCats.model")
    #X_test = CreateTestingData()
    #predictions = DC_model.predict(X_test)
    #print("X.shape = ", X_test.shape)
    #print("predictions.shape = ", predictions.shape)
    #print("predictions =\n", predictions)

    #plt.imshow(X_test[0], cmap = "gray")
    #plt.show()
    #print("Prediction = ", predictions[0])


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
            img_array = cv2.imread(os.path.join(TESTING_DIR, img), cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testing_data.append([resized_array])
        except Exception as e:
            print("Broken image. Passing")
            pass

    X = np.array(testing_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X/255.0
    return X

def PickleData(X):
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

def CreateModel(X, y, act, opt, batsz, ep, valSplit):

    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape= X.shape[1:]))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3)))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(X, y, batch_size=batsz, epochs=ep, validation_split=valSplit)
    return model


if __name__ == "__main__":
    main()
    
