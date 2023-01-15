"""encoder.py - Encodes image (Mel-scaled spectrogram) pixels into one-hot vectors."""

import numpy as np
import os
import json

from PIL import Image
from .view import Viewer


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


class BatchEncoder(Encoder):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.batch = os.listdir(image_dir)
        self.batch_size = len(self.batch)
        self.viewer = Viewer()

    def batch_encode(self, output_path="mel_encoded_batch-1.json"):
        encoded_images = []
        self.viewer.debug(f"Encoding batch to {output_path}...")
        output = {
            "batch_size": self.batch_size,
            "images": self.image_dir,
            "encodings": [],
        }

        for i, filename in enumerate(self.batch):
            self.viewer.info(f"Encoding image {i+1}/{self.batch_size}...")
            img = Image.open(os.path.join(self.image_dir, filename))
            pixels = np.array(img)
            encoder = Encoder(pixels)
            encoded_pixels = encoder.encode()
            self.viewer.debug(f"Encoded pixels --> {encoded_pixels}")
            encoded_images.append(encoded_pixels.tolist())

        assert len(encoded_images) == self.batch_size

        self.viewer.debug(f"Dumping jsonified encodings to {output_path}...")
        output["encodings"] = encoded_images
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

        return encoded_images


if __name__ == "__main__":
    encoder = BatchEncoder("mel/")
    encoded_images = encoder.batch_encode()
