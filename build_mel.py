import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Set the path to the directory containing the .wav files
wav_dir = "example-train/wav"

# Iterate over the .wav files in the directory
for wav_path in os.listdir(wav_dir):
    if wav_path.endswith(".wav"):
        # Load the waveform signal of the .wav file
        y, sr = librosa.load(os.path.join(wav_dir, wav_path))
        print(f"Bytes: {len(y)}")
        print(f"Sample Rate: {sr}")

        # Compute the mel-scaled spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

        # Convert the spectrogram from power to decibel (dB) units
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Display the mel spectrogram using a colormap
        plt.figure(figsize=(10, 4))
        mesh = librosa.display.specshow(
            S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000
        )
        plt.colorbar()
        plt.title("Mel-scaled Spectrogram")
        plt.tight_layout()

        # Save the mel spectrogram as a .png file
        filename = wav_path.split(wav_dir)[0].replace(".wav", ".png")
        print(filename)
        plt.savefig("mel/" + filename, dpi=300)
