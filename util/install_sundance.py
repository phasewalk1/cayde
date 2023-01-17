import subprocess

cmd = "cd sundance && cargo install --path ."
subprocess.run(cmd, shell=True)
