#!/usr/bin/env python3
"""Comment of Function"""
from tensorflow import keras as K


def lenet5(X):
    """LeNet-5 with Keras"""
    he_init = K.initializers.HeNormal(seed=0)
    conv1 = K.layers.Conv2D(
        6, (5, 5), padding='same', activation='relu',
        kernel_initializer=he_init
    )(X)
    pool1 = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(
        16, (5, 5), padding='valid', activation='relu',
        kernel_initializer=he_init
    )(pool1)
    pool2 = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    flat = K.layers.Flatten()(pool2)
    fc1 = K.layers.Dense(
        120, activation='relu', kernel_initializer=he_init
    )(flat)
    fc2 = K.layers.Dense(
        84, activation='relu', kernel_initializer=he_init
    )(fc1)
    output = K.layers.Dense(
        10, activation='softmax', kernel_initializer=he_init
    )(fc2)
    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer='adam', loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
