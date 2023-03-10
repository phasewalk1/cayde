"""preprocessor.py - Preprocessors and processing methods for generating supervised learning datasets."""

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json

from ..util.view import Viewer

# Set the sample rate for all files loaded from the DATASET_PATH
SAMPLE_RATE = 22050
# The duration of each .wav used in the dataset
DURATION = 30  # in seconds
# The number of samples per track
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Set STFT params
N_FFT = 2048
HOP_LENGTH = 512

# Set MFCC params
N_MFCC = 13

# Figure DPI
DPI = 100
# Figure Width
WIDTH = 1366
# Figure Height
HEIGHT = 96


class MelSpectrogramBuilder:
    def __init__(self, wav_dir, output_dir):
        self.output_dir = output_dir
        self.batch = os.listdir

    # compute the mel-scaled spectrogram for a single .wav file

    def single_mel(
        self, batch_mode=False, gtzan=False, wav_dir=None, show_waveform=False
    ):
        print(f"Loading {self.wav_path}...")
        if batch_mode:
            assert wav_dir is not None, "wav_dir must be specified in batch mode"

        song_name = self.get_song_name(self.wav_path)
        print(f"Computing mel-scaled spectrogram for {song_name}...")
        # load the waveform
        signal, sr = librosa.load(self.wav_path)
        if show_waveform:
            show_waveform(signal, sr)
        # compute the mel-scaled spectrogram
        S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        # construct the figure
        plt.figure(figsize=(8, 3), dpi=DPI)
        librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000)
        plt.colorbar()
        plt.title(f"Mel-scaled Spectrogram: {song_name}")
        plt.tight_layout()

        # save the figure
        if batch_mode and not gtzan:
            filename = self.wav_path.split(wav_dir)[-1].replace(".wav", ".png")
        elif batch_mode and gtzan:
            filename = self.get_song_name(self.wav_path) + ".png"
        else:
            # split at the last / to get the filename and remove the .wav extension
            filename = self.wav_path.split("/")[-1].replace(".wav", ".png")

        print(f"Saving mel-scaled spectrogram for {song_name} to {filename}...\n")
        plt.savefig("mel/" + filename, dpi=DPI)
        plt.close()
        plt.cla()
        plt.clf()

        return signal, sr

    # display the waveform
    def show_waveform(self, signal, sr):
        librosa.display.waveshow(signal, sr=sr)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    # get the song name from the path
    def get_song_name(self, wav_path):
        return wav_path.split("/")[-1].replace(".wav", "")

    # compute the mel-scaled spectrogram for all .wav files in a directory
    def batch_mel(self, wav_dir, gtzan=False):
        self.wav_dir = wav_dir
        # genre organized directory
        if gtzan:
            for genre in os.listdir(self.wav_dir):
                print("Looking for .wav files in " + genre + "...")
                for song in os.listdir(os.path.join(self.wav_dir, genre)):
                    self.wav_path = os.path.join(self.wav_dir, genre, song)
                    self.single_mel(batch_mode=True, gtzan=True, wav_dir=self.wav_dir)

        else:
            # Iterate over the .wav files in the directory
            for wav_path in os.listdir(wav_dir):
                if wav_path.endswith(".wav"):
                    wav_path = self.wav_dir + wav_path  # add the directory to the path
                    self.single_mel(batch_mode=True, gtzan=False, wav_dir=self.wav_dir)

    # compute the fft for a signal
    def compute_fft(signal):
        from numpy import fft

        fft_out = fft.fft(signal)
        magnitude = np.abs(fft_out)
        return magnitude

    # compute the spectrum for a fft-computed magnitude of a signal
    def fft_to_spectrum(magnitude, sr, show=False):
        frequency = np.linspace(0, sr, len(magnitude))
        left_freq = frequency[: int(len(frequency) / 2)]
        left_mag = magnitude[: int(len(magnitude) / 2)]
        if show:
            plt.cla()
            plt.clf()
            plt.plot(left_freq, left_mag)
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")
            plt.show()

    # compute the short-time fourier transform for a signal
    def compute_stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH):
        stft = librosa.core.stft(signal, n_fft=n_fft, hop_length=hop_length)
        return stft

    # log-scale the spectrogram
    def logscale(spectrogram):
        log_spec = librosa.amplitude_to_db(spectrogram)
        return log_spec

    # display a spectrogram
    def specshow(spectrogram, sr, hop_length=HOP_LENGTH, save=False, outpath=None):
        plt.clf()
        plt.cla()
        if save:
            assert outpath is not None, "filename must be specified in save mode"
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.colorbar()
            plt.savefig(outpath, dpi=300)
        else:
            librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.colorbar()
            plt.show()

    # save a spectrogram
    def specsave(self, spectrogram, sr, outpath, hop_length=HOP_LENGTH):
        self.specshow(
            spectrogram, sr, hop_length=hop_length, save=True, outpath=outpath
        )


class MFCCBuilder:
    def __init__(self, output_path):
        assert output_path.endswith(".json")
        self.output_path = output_path
        self.view = Viewer()

    # compute Mel-frequency cepstral coefficients (MFCCs)
    def compute_mfcc(signal, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC):
        MFCCs = librosa.feature.mfcc(
            signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc
        )
        return MFCCs

    def segmented_batch_save_mfcc(
        self,
        dataset_path,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_segments=5,
    ):
        """
        # Parameters:
            dataset_path (str): The path to the dataset containing audio files (.wav)
            json_path (str): The path to the output json that will store our labels and MFCCs
            n_mfcc (int, optional): Number of MFCCs to generate per segment. Defaults to N_MFCC (13).
            n_fft (int, optional): Number of FFT bands to compute. Defaults to N_FFT (2048).
            hop_length (int, optional): The linspace separator value for STFT. Defaults to HOP_LENGTH (512).
            win_func (int, optional): Which window function to apply. Defaults to np.hanning.
            n_segments (int, optional): Number of segments to split each audio file into. Defaults to 5.
        """

        # dictionary to store data
        extractings = {"mapping": [], "labels": [], "MFCCs": []}

        N_SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / n_segments)
        EXPECTED_MFCC_PER_SEGMENT = np.ceil(N_SAMPLES_PER_SEGMENT / hop_length)

        # loop through dataset that's organized by genre
        for genre_idx, (dirpath, _dirnames, filenames) in enumerate(
            os.walk(dataset_path)
        ):
            # ensures we're not at the root level
            if dirpath is not dataset_path:
                # save the semantic label
                dirpath_components = dirpath.split("/")
                semantic_label = dirpath_components[-1]
                extractings["mapping"].append(semantic_label)
                self.view.info(f"\nProcessing {semantic_label}")

                # process files for a specific genre
                for f in filenames:
                    # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # process segments extracting MFCCs and storing data
                    for s in range(n_segments):
                        # segment the signal into a slice of the audio file, this increases our training dataset size
                        start_sample = N_SAMPLES_PER_SEGMENT * s
                        finish_sample = start_sample + N_SAMPLES_PER_SEGMENT
                        # compute the MFCC matrix for the segment
                        mfcc = librosa.feature.mfcc(
                            y=signal[start_sample:finish_sample],
                            sr=sr,
                            n_fft=n_fft,
                            n_mfcc=n_mfcc,
                            hop_length=hop_length,
                        )
                        # transpose the mfcc matrix
                        MFCC = mfcc.T
                        # store the MFCCs for the segment if it has the expected length
                        if len(MFCC) == EXPECTED_MFCC_PER_SEGMENT:
                            extractings["MFCCs"].append(MFCC.tolist())
                            extractings["labels"].append(genre_idx - 1)
                            self.view.debug(f"{file_path}, segment: {s}")

        # save MFCCs to json file
        with open(self.output_path, "w") as out:
            json.dump(extractings, out, indent=4)

    # display MFCCs
    def show_mfcss(MFCCs, sr, hop_length=HOP_LENGTH, save=False, outpath=None):
        plt.clf()
        plt.cla()

        if save:
            assert outpath is not None, "filename must be specified in save mode"
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
            plt.xlabel("Time")
            plt.ylabel("MFCC")
            plt.colorbar()
            plt.savefig(outpath, dpi=300)
        else:
            librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
            plt.xlabel("Time")
            plt.ylabel("MFCC")
            plt.colorbar()
            plt.show()
