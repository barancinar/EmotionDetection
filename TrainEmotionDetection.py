import numpy as np
import tensorflow as tf
import cv2
import os

trainPath = "C:/Users/cinar/PycharmProjects/EmotionDetection/data/train"
testPath = "C:/Users/cinar/PycharmProjects/EmotionDetection/data/test"

folderList = os.listdir(trainPath)
folderList.sort()

print(folderList)

x_train = []
y_train = []

x_test = []
y_test = []

# load the train data info arrays
for i, category in enumerate(folderList):
    files = os.listdir(trainPath + "/" + category)
    for file in files:
        # print(category + "/" + file)
        img = cv2.imread(trainPath + "/" + category + "/{0}".format(file), 0)
        x_train.append(img)
        y_train.append(i)

print(len(x_train))  # 28709
print(len(y_train))

folderList = os.listdir(testPath)
folderList.sort()

# load the test data info arrays
for i, category in enumerate(folderList):
    files = os.listdir(testPath + "/" + category)
    for file in files:
        # print(category + "/" + file)
        img = cv2.imread(testPath + "/" + category + "/{0}".format(file), 0)
        x_test.append(img)
        y_test.append(i)

print("Test data: ")
print(len(x_test))  # 7128
print(len(y_test))

# convert the data to array

x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

# check
print(x_train.shape)
print(x_train[0])

# 2 tasks
# normalize the image: 0 to 1
# add another dimention to the data: (28709, 48, 48, 1)

x_train = x_train / 255.0
x_test = x_test / 255.0

# reshape the train data

numOfImages = x_train.shape[0]
x_train = x_train.reshape(numOfImages, 48, 48, 1)

print(x_train[0])
print(x_train.shape)

# reshape the test data

numOfImages = x_test.shape[0]
x_test = x_test.reshape(numOfImages, 48, 48, 1)

print(x_test.shape)

# convert the labels to categorical
from keras.utils import to_categorical  # type: ignore

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

print("to categorical:")
print(y_train)
print(y_train.shape)
print(y_train[0])

# build the model :
# ====================

input_shape = x_train.shape[1:]
print(input_shape)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

model = Sequential()

model.add(Input(shape=input_shape))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dense(7, activation="softmax"))

print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

batch = 32
epochs = 30

stepsPerEpoch = np.ceil(len(x_train) / batch)
validationSteps = np.ceil(len(x_test) / batch)

stopEarly = EarlyStopping(monitor='val_accuracy', patience=5)

# train the model
history = model.fit(x_train,
                    y_train,
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=[stopEarly])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# show the charts

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label="Train Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Trainig and Validation Accuracy")
plt.legend(loc='lower right')
plt.show()

plt.plot(epochs, acc, 'r', label="Train Loss")
plt.plot(epochs, val_acc, 'b', label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Trainig and Validation Loss")
plt.legend(loc='upper right')
plt.show()

# save the model

modelFileName = "C:/Users/cinar/PycharmProjects/EmotionDetection/model/emotion.h5"
model.save(modelFileName)
