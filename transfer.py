# codeReview3.py
# Anselmis Columna & Sebastian Paquette
# Dr. Eicholtz
# 25 April 2023
# CSC 4510
# transfer.py
# Apply transfer learning to Xception net in order to predict which computer science professor is
# most likely to wear a given shirt.
#
# References:
# 1. https://arxiv.org/pdf/1610.02357.pdf
# 2. https://keras.io/api/applications/xception/
# 3. https://keras.io/guides/transfer_learning/

import tensorflow as tf
from tensorflow import keras
import keras.models
import keras.optimizers
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.util import montage
from skimage.color import label2rgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pdb

# The data needs to be in a folder called 'Data' in the same directory as the python file for this to work
DATAROOT = f'{os.path.dirname(__file__)}\Data'


def main():
    # Load dataset
    # pdb.set_trace()
    data, labels = load(DATAROOT, show=False)
    labels, y = np.unique(labels, return_inverse=True)

    # TODO: Split data into random shuffled training (75%) and testing (25%) sets
    xtrain, xtest, ytrain, ytest = train_test_split(data, y, train_size=0.75)

    # TODO: Preprocess image data to match Xception requirements
    # (see keras.applications.xception.preprocess_input)
    xtrain = keras.applications.xception.preprocess_input(xtrain)
    xtest = keras.applications.xception.preprocess_input(xtest)

    # TODO: Load the Xception model from Keras, use weights from "imagenet" and do not
    # include the top (output) layer
    xception = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet'
        )

    # Change the end of the network to match new dataset (this is what will be learned!)
    avg = keras.layers.GlobalAveragePooling2D()(xception.output)
    output = keras.layers.Dense(len(labels), activation='softmax')(avg)
    model = keras.Model(inputs=xception.inputs, outputs=output) # TODO: Make a new model with the old xception input and the new dense output
    model.summary()

    # TODO: Freeze layers in xception
    for layer in xception.layers:
        layer.trainable = False

    # Compile the model
    # TODO: Use the SGD optimizer with learning rate of 0.1 and momentum of 0.9
    optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Create callbacks
    filename = os.path.join("transfer.h5")
    checkpoint = keras.callbacks.ModelCheckpoint(filename, save_best_only=True)
    earlystopping =  keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # TODO: Train the model using the fit method; use a batch size of 5, 100 epochs, the callbacks created above, and the test data for validation
    history = model.fit(xtrain, ytrain, 
                        batch_size=5, epochs=100, 
                        callbacks=[checkpoint, earlystopping], 
                        validation_data=(xtest, ytest)
                        )

    # Evaluate the model
    scores = model.evaluate(xtest, ytest)

    # TODO: Compute the confusion matrix and show which shirts were misclassified
    pred = model.predict(xtest)
    y_classes = pred.argmax(axis=-1)
    matrix = confusion_matrix(ytest, y_classes)

    bad = np.where(y_classes != ytest)

    plt.figure(2)
    m = montage(data[bad], padding_width=1, multichannel=True)
    plt.imshow(m)
    plt.title('Misclassified Images')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    #Debug if necessary
    # pdb.set_trace()
    

    


def load(directory=DATAROOT, show=False):
    '''Load data (and labels) from directory.'''
    files = os.listdir(directory)  # extract filenames
    n = len(files)  # number of files
    sz = [224, 224]  # required size of images for network

    # Read images from file and store in data array
    x = np.zeros([n] + sz + [3], dtype=int)
    for i in range(n):
        print(f'Loading data from file: {files[i]}')
        img = imread(os.path.join(directory, files[i]))
        if img.dtype == 'float32':
            img *= 255
        x[i] = np.array(tf.image.resize(img, sz))

    # Show montage of images (optional)
    if show:
        plt.figure(1)
        m = montage(x, fill=np.zeros(3), padding_width=1, multichannel=True)
        plt.imshow(m)
        plt.title('Sample Images')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    # Extract labels from filenames
    y = np.array([file.split('.')[0][:-2] for file in files])

    return x, y


if __name__ == '__main__':
    main()
