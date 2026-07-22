import tensorflow as tf
from tensorflow import keras
from keras import layers

def get_model(s):
    inputs = keras.Input(shape=(s,s,20), name = 'The game map')
    f = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    f = layers.Conv2D(64, 3, padding="same", activation="relu")(f)
    f = layers.Conv2D(64, 3, padding="same", activation="relu")(f)
    # f = layers.Conv2D(128, 3, padding="same", activation="relu")(f)
    units = layers.Conv2D(6, 1, activation="linear")(f)
    cities = layers.Conv2D(2, 1, activation="linear")(f)
    output = layers.Concatenate()([units, cities])
    # Compiling happens in training, not here
    # model = keras.Model(inputs, output)
    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
    #     loss=tf.keras.losses.Huber()
    # )
    # return model
    return keras.Model(inputs, output)
