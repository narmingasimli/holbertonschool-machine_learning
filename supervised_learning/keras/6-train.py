#!/usr/bin/env python3
"""Outside of Function"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Train Model"""
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,))
    History = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle,
                          callbacks=callbacks)
    return History
