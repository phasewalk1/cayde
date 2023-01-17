import json
import numpy as np
import os
import re
import matplotlib.pyplot as plt

from util.view import Viewer
from tensorflow import keras
from sklearn.model_selection import train_test_split


def load_data(processed_dataset_path="data.json"):
    print("\nLoading data...")
    with open(processed_dataset_path, "r") as fp:
        data = json.load(fp)
    inputs = np.array(data["MFCCs"])
    targets = np.array(data["labels"])

    return inputs, targets


class Trainer:
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.view = Viewer()

    def split_data(self, train_size=0.8, test_size=0.2):
        self.view.info("\nSplitting data...")
        if train_size + test_size != 1:
            raise ValueError("Train size and test size must add up to 1.")
        x_train, x_test, y_train, y_test = train_test_split(
            self.inputs,
            self.targets,
            test_size=test_size,
        )

        return x_train, x_test, y_train, y_test

    def find_latest_ckpt(self, ckpt_dir):
        checkpoints = []
        for filename in os.listdir(ckpt_dir):
            if filename.startswith("ckpt"):
                checkpoints.append(filename)
        for ckpt in sorted(checkpoints):
            ckpt_num = int(re.findall(r"\d+", ckpt)[0])
        self.view.info(f"Latest checkpoint: {ckpt_num} found.")
        new_ckpt_num = ckpt_num
        return new_ckpt_num

    def train(
        self,
        epochs=10,
        batch_size=64,
        learning_rate=0.0001,
        keras_mode=False,
        ckpt_mode=False,  # load from latest checkpoint before training
        ckpt_path=None,
    ):
        x_train, x_test, y_train, y_test = self.split_data(
            train_size=0.7, test_size=0.3
        )

        if keras_mode:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            if ckpt_mode and ckpt_path is None:
                raise ValueError("ckpt_path must be specified in ckpt_mode.")
            elif ckpt_mode:
                self.view.debug(f"Loading model from checkpoint...{ckpt_path}")
                self.model.load_weights(ckpt_path)

            self.model.fit(
                x_train,
                y_train,
                validation_data=(x_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
            )

            new_ckpt = self.find_latest_ckpt("checkpoints") + 1
            self.view.debug(f"Saving model to checkpoint...{new_ckpt}")
            self.model.save_weights(f"checkpoints/ckpt-{new_ckpt}")

            self.view.info("Plotting model performance...")
            plt.plot(self.model.history.history["accuracy"], label="train accuracy")
            plt.plot(self.model.history.history["val_accuracy"], label="test accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(loc="lower right")
            plt.title(f"Model performance - ckpt{new_ckpt}")
            plt.savefig(f"perf/model_performance-ckpt{new_ckpt}.png")
        else:
            # TODO: Implement statistics tracking/plotting for non-Keras models.
            for epoch in range(epochs):
                for batch in range(len(x_train) // batch_size):
                    batch_x = x_train[batch * batch_size : (batch + 1) * batch_size]
                    batch_y = y_train[batch * batch_size : (batch + 1) * batch_size]

                    x = self.model.forward(batch_x)
                    self.model.optimizer.zero_grad()
                    loss = self.model.loss(x, batch_y)
                    loss.backward()
                    self.model.optimizer.step()

                print(f"Epoch {epoch + 1} complete.")

        plt.cla()
        plt.clf()
