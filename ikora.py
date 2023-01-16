import click

from wrangling.wrangler import Wrangler, WMODES as MODES
from wrangling.fetcher import FFLAGS


@click.command()
@click.option(
    "--mode",
    type=click.Choice(MODES),
    required=True,
    help="Choose 'mfcc' to preprocess the data into MFCCs, or 'encode' to encode the images into one-hot encodings.",
)
@click.option(
    "-x",
    type=click.Choice(FFLAGS),
    required=False,
)
def main(mode, x=None):
    print(f"Running ikora.py in {mode}:[{x}] mode...")

    wrangler = Wrangler(mode=mode, flags=x)
    wrangler.preprocess()


if __name__ == "__main__":
    main()
