#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Function  that creates an autoencoder"""
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
    encoder = keras.Model(inputs=encoder_input, outputs=latent)

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for units in hidden_layers[::-1]:
        x = keras.layers.Dense(units, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output)

    auto_input = encoder_input
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(inputs=auto_input, outputs=auto_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
