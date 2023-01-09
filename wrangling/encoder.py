"""encoder.py - Encodes image (Mel-scaled spectrogram) pixels into one-hot vectors."""

import numpy as np
import os
from PIL import Image


class Encoder:
    def __init__(self, pixels):
        self.pixels = pixels

    def encode(self):
        one_hot_encoded = []

        for row in self.pixels:
            one_hot_row = []
            for pixel in row:
                if np.any(pixel == 0):
                    one_hot_vector = [1, 0]
                else:
                    one_hot_vector = [0, 1]
                one_hot_row.append(one_hot_vector)
            one_hot_encoded.append(one_hot_row)

        one_hot_encoded = np.array(one_hot_encoded)
        print(one_hot_encoded.shape)

        self.encoded_pixels = one_hot_encoded
        return self.encoded_pixels


def batch_encode(image_dir):
    encoded_pixels = np.array([])
    for image in os.listdir(image_dir):
        if image.endswith(".png"):
            image_path = image_dir + image
            image = Image.open(image_path)
            pixels = np.array(image)
            encoder = Encoder(pixels)
            encoded_pixels = np.append(encoded_pixels, encoder.encode())
    return encoded_pixels
