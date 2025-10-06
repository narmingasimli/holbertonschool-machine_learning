#!/usr/bin/env python3
import numpy as np

def preprocess_data(X, Y):
    """Comment of Function"""
    X_resized = np.array([tf.image.resize(img, (160, 160)).numpy() for img in X])
    X_p = K.applications.mobilenet_v2.preprocess_input(X_resized)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def build_base_model():
    """Comment of Function"""
    base_model = K.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    return base_model

def build_head_model(input_shape):
    
    inputs = K.Input(shape=input_shape)
    x = K.layers.GlobalAveragePooling2D()(inputs)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    model = K.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    # Load CIFAR-10 data
    (X_train, Y_train), (X_val, Y_val) = K.datasets.cifar10.load_data()
    Y_train = Y_train.flatten()
    Y_val = Y_val.flatten()

    # Preprocess data
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_val_p, Y_val_p = preprocess_data(X_val, Y_val)

    # Build and freeze base model
    base_model = build_base_model()

    # Precompute bottleneck features
    print("Calculating bottleneck features for training data...")
    bottleneck_train = base_model.predict(X_train_p, batch_size=64, verbose=1)
    print("Calculating bottleneck features for validation data...")
    bottleneck_val = base_model.predict(X_val_p, batch_size=64, verbose=1)

    # Build and compile head model
    head_model = build_head_model(bottleneck_train.shape[1:])

    # Train only the head
    head_model.fit(
        bottleneck_train, Y_train_p,
        batch_size=64,
        epochs=30,
        validation_data=(bottleneck_val, Y_val_p),
        verbose=2
    )

    # Save the entire model (head + base)
    # To do this, create a combined model that includes the base and head
    inputs = K.Input(shape=(160,160,3))
    x = base_model(inputs, training=False)  # base frozen
    outputs = head_model(x)
    full_model = K.Model(inputs, outputs)

    # Compile full model
    full_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    full_model.save('cifar10.h5')
    print("Model saved as cifar10.h5")

if __name__ == '__main__':
    train_model()
