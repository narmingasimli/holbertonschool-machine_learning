#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Function that creates a convolutional autoencoder"""
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)

    encoder = keras.Model(inputs=encoder_input, outputs=x)

    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    for i, f in enumerate(filters[::-1]):
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                padding='same'
                                if i < len(filters) - 1 else 'valid',
                                activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    decoder_output = keras.layers.Conv2D(filters=input_dims[-1],
                                         kernel_size=(3, 3),
                                         padding='same',
                                         activation='sigmoid')(x)

    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output)

    auto_input = encoder_input
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(inputs=auto_input, outputs=auto_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
