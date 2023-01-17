import subprocess
import os
import click
import shutil

from .view import Viewer


FFLAGS = ["clean", "full", "reduced"]


class Fetcher:
    DATASET = "andradaolteanu/gtzan-dataset-music-genre-classification"
    UNZIP_DIR = "example-train/GTZAN-Full"
    DATA_DIR = UNZIP_DIR + "/Data"
    SOURCE_DIR = UNZIP_DIR + "/Data/genres_original"
    CMD = f"kaggle datasets download -d {DATASET} --unzip --path {UNZIP_DIR}"

    def __init__(self, fflag="full"):
        self.viewer = Viewer()
        self.fflag = fflag

    def fetch(self):
        if os.path.exists(self.UNZIP_DIR):
            shutil.rmtree(self.UNZIP_DIR)
        self.viewer.info(f"Fetching dataset...")
        output = subprocess.run(self.CMD, shell=True, capture_output=True)
        self.viewer.info(f"Dataset fetched!: {output.stdout.decode()}")

    def clean(self):
        self.viewer.info("Cleaning dataset...")

        if self.fflag == "full":
            if os.path.exists(self.DATA_DIR):
                for genre in os.listdir(self.SOURCE_DIR):
                    new_genre_dir = os.path.join("example-train/GTZAN-Full", genre)
                    self.viewer.debug(f"Creating {new_genre_dir}...")
                    os.mkdir(new_genre_dir)
                    self.viewer.debug(f"Looking at {genre}...")
                    for file in os.listdir(os.path.join(self.SOURCE_DIR, genre)):
                        file_path = os.path.join(self.SOURCE_DIR, genre, file)
                        self.viewer.debug(f"Moving {file_path} to {new_genre_dir}..")
                        shutil.move(file_path, new_genre_dir)
            else:
                self.viewer.info("Dataset already cleaned!")

        # Remove corrupted file
        CORRUPTED = self.UNZIP_DIR + "/jazz/jazz.00054.wav"
        os.remove(CORRUPTED)

        shutil.rmtree("example-train/GTZAN-Full/Data")
        self.viewer.info("Dataset cleaned!")
