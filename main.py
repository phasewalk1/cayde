from models.kgcmfcc import KerasGenreClassifier
from trainer import Trainer, load_data


PROCESSED_DATASET = "data.json"
EPOCHS = 50
BATCH_SIZE = 128


def main():
    inputs, targets = load_data(processed_dataset_path=PROCESSED_DATASET)
    model = KerasGenreClassifier(inputs)
    trainer = Trainer(model, inputs, targets)

    # train indefinitely and save checkpoints
    while True:
        latest_ckpt = trainer.find_latest_ckpt(ckpt_dir="checkpoints")
        print(f"Training from checkpoint {latest_ckpt}...")
        trainer.train(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=0.0001,
            keras_mode=True,
            ckpt_mode=True,
            ckpt_path=f"checkpoints/ckpt-{latest_ckpt}",
        )


if __name__ == "__main__":
    main()
