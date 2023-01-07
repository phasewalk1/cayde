import numpy as np
from PIL import Image

image = Image.open("sample/mel_spec.png")
pixels = np.array(image)

one_hot_encoded = []

for row in pixels:
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