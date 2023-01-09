from model import KerasGenreClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
from tensorflow import keras

# Procesed dataset path
PROCESSED_DATASET = "data.json"

# Hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.0001
BATCH_SIZE = 128


def load_data(processed_dataset_path=PROCESSED_DATASET):
    with open(processed_dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["MFCCs"])
    targets = np.array(data["labels"])

    return inputs, targets


def split_data(inputs, targets, train_size=0.8, test_size=0.2):
    if train_size + test_size != 1:
        raise ValueError("Train size and test size must add up to 1.")
    x_train, x_test, y_train, y_test = train_test_split(
        inputs,
        targets,
        test_size=test_size,
    )

    return x_train, x_test, y_train, y_test


def load_from_ckpt(model, ckpt_path):
    model.load_weights(ckpt_path)
    return model


def find_latest_ckpt(ckpt_dir):
    checkpoints = []
    for filename in os.listdir(ckpt_dir):
        if filename.startswith("ckpt"):
            checkpoints.append(filename)
    for ckpt in sorted(checkpoints):
        ckpt_num = int(re.findall(r"\d+", ckpt)[0])
    print(f"Latest checkpoint: {ckpt_num} found.")
    new_ckpt_num = ckpt_num
    return new_ckpt_num


# train the keras classifier model
def train_keras_classifier(ckpt_mode=False, ckpt_path=None):
    print("\nLoading data...")
    inputs, targets = load_data()
    print(f"Input shapes: {inputs.shape[1]} x {inputs.shape[2]}")
    print("Data loaded.\nSplitting data...")
    x_train, x_test, y_train, y_test = split_data(
        inputs, targets, train_size=0.7, test_size=0.3
    )

    # build and compile the model
    model = KerasGenreClassifier(inputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if ckpt_mode and ckpt_path is None:
        raise ValueError("ckpt_path must be specified in ckpt_mode.")
    elif ckpt_mode:
        print(f"Loading model from checkpoint...{ckpt_path}")
        model.load_weights(ckpt_path)
        print(f"Model loaded from checkpoint...{ckpt_path}")

    # train the model
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # save the models weights as a checkpoint
    new_ckpt = find_latest_ckpt("checkpoints") + 1
    print(f"Saving model as checkpoint {new_ckpt}...")

    model.save_weights(f"checkpoints/ckpt-{new_ckpt}")

    # plot the models performance on the validation set
    print("Plotting model performance...")
    plt.plot(model.history.history["accuracy"], label="train accuracy")
    plt.plot(model.history.history["val_accuracy"], label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.title(f"Model performance - ckpt{new_ckpt}")
    plt.legend()
    plt.savefig(f"perf/model_performance-ckpt{new_ckpt}.png")


if __name__ == "__main__":
    latest_ckpt = find_latest_ckpt("checkpoints")
    train_keras_classifier(ckpt_mode=True, ckpt_path=f"checkpoints/ckpt-{latest_ckpt}")
