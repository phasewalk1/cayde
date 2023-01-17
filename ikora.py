import click
import subprocess

from wrangling.wrangler import Wrangler
from wrangling.fetcher import FFLAGS


MODES = ["mfcc", "encode", "fetch", "sundance"]
SUNDANCE_OPTS = ["run", "install"]

X = FFLAGS + SUNDANCE_OPTS


@click.command()
## IKORA MODE!!
@click.option(
    "--mode",
    type=click.Choice(MODES),
    required=True,
    help="Choose 'mfcc' to preprocess the data into MFCCs, or 'encode' to encode the images into one-hot encodings.",
)
## FFLAGS UNIQUE TO MODES
@click.option(
    "-x",
    type=click.Choice(X),
    required=False,
)
def main(mode, x=None):
    print(f"Running ikora.py in {mode}:[{x}] mode...")

    if mode != "sundance":
        wrangler = Wrangler(mode=mode, flags=x)
        wrangler.preprocess()
    else:
        if x is None or x == "run":
            cmd = "cd sundance && cargo run -- -d ../mel/"
            subprocess.run(cmd, shell=True)
        elif x == "install":
            cmd = "cd sundance && cargo install --path ."
            subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
