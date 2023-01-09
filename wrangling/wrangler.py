"""wrangler.py - Uses defined preprocessors to wrangle the data into a format that can be used to train the model."""

from preprocessors import MFCCBuilder
from encoder import BatchEncoder
import sys

DATASET = "example-train/GTZAN-Full"
OUTPUT_FILE = "data.json"


class Wrangler:
    def __init__(self, mode):
        self.mode = mode

    def preprocess(self):
        if self.mode == "mfcc":
            builder = MFCCBuilder(output_path=OUTPUT_FILE)
            builder.segmented_batch_save_mfcc(dataset_path=DATASET)
        elif self.mode == "encode":
            encoder = BatchEncoder(image_dir=DATASET)
            encoder.batch_encode(output_path=OUTPUT_FILE)
        else:
            raise ValueError("Invalid mode. Please choose 'mfcc' or 'encode'.")


# Currently, this script acts to preprocess the reduced GTZAN dataset into a JSON file that contains mappings, labels, and MFCCs
# used to supervise the genre classifier model.
if __name__ == "__main__":
    mode = sys.argv[1]
    wrangler = Wrangler(mode=mode)
    wrangler.preprocess()
