import librosa
import librosa.display
import time
import numpy as np
from matplotlib import pyplot as plt


def load_wav(path):
    data, sample_rate = librosa.load(path, res_type='kaiser_best')
    return data, sample_rate


def to_mel_spec(data, sr=44100, n_mels=128, fmax=8000):
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def build_mesh_img(mel_spec_db, sr=44100, x_axis="time", y_axis="mel"):
    mesh = librosa.display.specshow(mel_spec_db, sr=sr, x_axis=x_axis, y_axis=y_axis)
    return mesh


def make_save_mesh(mesh):
    plt.colorbar(mesh, format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.savefig("sample/mel_spec.png", bbox_inches="tight")


def main():
    start_time = time.perf_counter()

    data, sr = load_wav("sample/test.wav")
    mel_spec_db = to_mel_spec(data, sr=sr, n_mels=128, fmax=8000)
    mesh = build_mesh_img(mel_spec_db, sr=sr)
    end_time = time.perf_counter()

    make_save_mesh(mesh)

    bench = end_time - start_time
    return bench


if __name__ == "__main__":
    exec_time = main()
    print(f"Execution time: {exec_time:.2f} seconds")
