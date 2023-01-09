"""wrangler.py - Uses defined preprocessors to wrangle the data into a format that can be used to train the model."""

from preprocessors import batch_save_mfcc

DATASET = "example-train/GTZAN-Full"
OUTPUT_DIR = "data.json"

# Currently, this script acts to preprocess the reduced GTZAN dataset into a JSON file that contains mappings, labels, and MFCCs
# used to supervise the genre classifier model.
if __name__ == "__main__":
    batch_save_mfcc(
        dataset_path=DATASET,
        json_path=OUTPUT_DIR,
    )
