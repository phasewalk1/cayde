"""wrangler.py - Uses defined preprocessors to wrangle the data into a format that can be used to train the model."""

from .preprocessors import MFCCBuilder
from .encoder import BatchEncoder
from .fetcher import Fetcher

DATASET = "example-train/GTZAN-Reduced"
OUTPUT_FILE = "data.json"

IMG_DIR = "mel/"
ENCODINGS_OUT = "mel_encoded_batch-1.json"


class Wrangler:
    def __init__(self, mode, flags=None):
        self.mode = mode
        if flags is not None:
            self.flags = flags

    def preprocess(self):
        if self.mode == "mfcc":
            builder = MFCCBuilder(output_path=OUTPUT_FILE)
            builder.segmented_batch_save_mfcc(dataset_path=DATASET)
        elif self.mode == "encode":
            encoder = BatchEncoder(image_dir=IMG_DIR)
            encoder.batch_encode(output_path=ENCODINGS_OUT)
        elif self.mode == "fetch":
            fetcher = Fetcher()
            if self.flags == "clean":
                fetcher.clean()
            elif self.flags == "full":
                fetcher.fetch()
                fetcher.clean()
            else:
                fetcher.fetch()
        else:
            raise ValueError("Invalid mode. Please choose 'mfcc' or 'encode'.")
