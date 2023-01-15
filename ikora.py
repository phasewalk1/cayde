import click
import sys

from wrangling.wrangler import Wrangler
from wrangling.view import Viewer


MODES = ["mfcc", "encode"]


@click.command()
@click.option(
    "--mode",
    type=click.Choice(MODES),
    required=True,
    help="Choose 'mfcc' to preprocess the data into MFCCs, or 'encode' to encode the images into one-hot encodings.",
)
def main(mode):
    view = Viewer()

    if mode not in MODES or mode is None:
        view.critical("Invalid mode. Please choose 'mfcc' or 'encode'.")
        sys.exit(1)
    else:
        view.info(f"Running ikora.py in {mode} mode...")

        wrangler = Wrangler(mode=mode)
        wrangler.preprocess()


if __name__ == "__main__":
    main()
