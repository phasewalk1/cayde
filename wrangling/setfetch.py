import subprocess

DATASET = "andradaolteanu/gtzan-dataset-music-genre-classification"
cmd = f"kaggle datasets download -d {DATASET} --unzip --path example-train/GTZAN-Full"

print("Downloading dataset...")
output = subprocess.run(cmd, shell=True, capture_output=True)
print(output.stdout.decode())
