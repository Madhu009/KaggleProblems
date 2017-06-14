# coding: utf-8

import numpy as np

filepath = 'C:/Users/Madhu/Desktop/kag/tr.csv'
np.random.seed(7)

def getdata(filepath):
    X = []
    Y = []

    header = True
    for line in open(filepath):

        if header:
            header = False
        else:
            row = line.split(",")
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    return X, Y

X, Y = getdata(filepath)
print(X.shape)
print(Y.shape)

num_classes = len(set(Y))

def classes_count(Y):
    num_classes = set(Y)
    classes = [0, 0, 0, 0, 0, 0, 0]

    for i in range(len(num_classes)):
        count = 0
        for r in Y:
            if r == i:
                count += 1
        classes[i] = count
    return classes


classes = classes_count(Y)

# reshape X to fit keras with tensorflow backend
N, D = X.shape
X = X.reshape(N, 48, 48, 1)  # last dimension =1 is because it is black and white image, if colored, it will be 3

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

Y_train = (np.arange(num_classes) == Y_train[:, None]).astype(np.float32)
Y_test = (np.arange(num_classes) == Y_test[:, None]).astype(np.float32)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras import backend as K

K.set_image_dim_ordering('tf')


def createmodel(batch_size, epochs):
    model = Sequential()
    # Step 1 - Convolution
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(48, 48, 1), activation='relu'))

    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size=(5, 5)))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))

    # Step 3 - Flattening
    model.add(Flatten())
    # Step 4 - Full connection
    model.add(Dense(output_dim=512, activation='relu'))  # the output_dim is chosen by experience
    model.add(Dense(output_dim=512, activation='relu'))
    model.add(Dense(output_dim=7, activation='sigmoid'))

    # Compiling the CNN
    sgd = SGD(lr=10e-5, momentum=0.99, decay=0.9999, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=20,
              verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return model


model = createmodel(30, 25)

# Save the model
jsonFile = model.to_json()
with open('C:/Users/Madhu/Desktop/KerasModel/CNN/Facial_Expression/object1.json', 'w') as file:
    file.write(jsonFile)
model.save_weights('C:/Users/Madhu/Desktop/KerasModel/CNN/Facial_Expression/object1.h5')

print("training is done")


