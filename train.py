from model import build_keras_genre_classifier
from sklearn.model_selection import train_test_split
import numpy as np
import json
import matplotlib.pyplot as plt

# Procesed dataset path
PROCESSED_DATASET = "data.json"

# Hyperparameters
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32


def load_data(processed_dataset_path=PROCESSED_DATASET):
    with open(processed_dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["MFCCs"])
    targets = np.array(data["labels"])

    return inputs, targets


def split_data(inputs, targets, train_size=0.8, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(
        inputs,
        targets,
        test_size=test_size,
    )

    return x_train, x_test, y_train, y_test


def train():
    inputs, targets = load_data()
    x_train, x_test, y_train, y_test = split_data(inputs, targets)

    model = build_keras_genre_classifier(inputs)

    model.summary()
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=50,
        batch_size=32
    )

    # plot the results of training over time
    plt.plot(model.history.history["accuracy"], label="train accuracy")
    plt.plot(model.history.history["val_accuracy"], label="test accuracy")
    plt.legend()
    plt.show()

    model.save_weights("checkpoints/ckpt1")


if __name__ == "__main__":
    train()