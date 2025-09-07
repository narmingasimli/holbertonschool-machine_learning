#!/usr/bin/env python3
"""Outside of Function"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """Train Model"""
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,))

    if learning_rate_decay and validation_data is not None:
        def decay(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(decay, verbose=1))

    if save_best and validation_data is not None:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath=filepath, monitor='val_loss',
            save_best_only=save_best
        ))
    History = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle,
                          callbacks=callbacks)
    return History
