import os

import numpy as np
import soundfile as sf
import librosa

from scipy import signal


def read_wav_file(filename):
    wav, sample_rate = sf.read(filename)
    return wav, sample_rate


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = signal.get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)


def build_mel_basis(sample_rate=16000, n_fft=1024, n_mels=80, fmin=90, fmax=7600):
    return librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T


def build_min_level():
    return np.exp(- 100 / 20 * np.log(10))


mel_basis = build_mel_basis(sample_rate=16000, n_fft=1024, n_mels=80, fmin=90, fmax=7600)
min_level = build_min_level()
b, a= butter_highpass(cutoff=30, fs=16000, order=5)


def preprocess(inputs):
    if isinstance(inputs, np.ndarray):
        wav = inputs
    elif os.path.isfile(inputs):
        wav, sample_rate = read_wav_file(inputs)
    else:
        raise ValueError("Out of supported format")

    wav = signal.filtfilt(b=b, a=a, x=wav)

    D = pySTFT(wav).T

    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)

    return S


#spectro = preprocess(r"D:\VCTK_Corpus\VCTK-Corpus\wav48\p225\p225_001.wav")
#print(spectro.shape)