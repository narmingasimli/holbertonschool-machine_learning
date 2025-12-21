#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Function that creates a variational autoencoder"""
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = (keras.backend.
                   random_normal(shape=keras.backend.shape(z_mean)))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(inputs=encoder_input, outputs=[z, z_mean, z_log_var])

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for units in hidden_layers[::-1]:
        x = keras.layers.Dense(units, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output)

    auto_input = encoder_input
    z_sample, z_mean_sample, z_log_var_sample = encoder(auto_input)
    auto_output = decoder(z_sample)
    auto = keras.Model(inputs=auto_input, outputs=auto_output)

    reconstruction_loss = keras.losses.binary_crossentropy(auto_input,
                                                           auto_output)
    reconstruction_loss *= input_dims
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var_sample - keras.backend.square(z_mean_sample)
        - keras.backend.exp(z_log_var_sample), axis=-1
    )
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')
    return encoder, decoder, auto
