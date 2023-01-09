"""kgcmfcc.py - The Keras implementation of the Genre Classifier model using Mel-frequency cepstral coefficients (MFCCs) as input."""

from tensorflow import keras


"""The same model architecture as 'models/gcmfcc.py', but this time implemented using Keras."""


class KerasGenreClassifier(keras.Model):
    def __init__(self, inputs):
        super(KerasGenreClassifier, self).__init__()
        self.flatten = keras.layers.Flatten(
            input_shape=(inputs.shape[1], inputs.shape[2])
        )
        self.fc1 = keras.layers.Dense(512, activation="relu")
        self.fc2 = keras.layers.Dense(256, activation="relu")
        self.fc3 = keras.layers.Dense(64, activation="relu")
        self.fc4 = keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)
