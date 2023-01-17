import subprocess
import os


pyfmt = "black ."
rsfmt = "cd sundance && cargo fmt --all && cd .."

subprocess.run(pyfmt, shell=True)
subprocess.run(rsfmt, shell=True)
