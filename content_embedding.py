import os
import math

import numpy as np
import librosa

import torch
import torchvision

from resemblyzer import VoiceEncoder, preprocess_wav

from speakers_embedding import compute_speakers_embedding
from preprocess import preprocess

from model_vc import Generator, Encoder


def get_all_subdirs(dir):
    list_dir = []
    for dirs, _, files in os.walk(dir):
        subdirs = os.listdir(dirs)
        is_subdir = True
        for subdir in subdirs:
            if os.path.isdir(os.path.join(dirs, subdir)):
                is_subdir = False
                break
        if is_subdir:
            list_dir.append(dirs)

    return list_dir


def get_all_files_in_subdir(dir):
    list_file = []
    files = os.listdir(dir)
    for file in files:
        assert os.path.isfile(os.path.join(dir, file))
        list_file.append(os.path.join(dir, file))

    return list_file


if __name__ == '__main__':

    list_dir = get_all_subdirs(r"D:\VCTK_Corpus\VCTK-Corpus\wav48")
    print(list_dir)
    window_size = 0.3

    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    G = Generator(32, 256, 512, 32).eval().to(device)

    if torch.cuda.is_available():
        g_checkpoint = torch.load(r"E:\PythonProjects\Face_Animation\autovc\autovc.ckpt")
    else:
        g_checkpoint = torch.load(r"E:\PythonProjects\Face_Animation\autovc\autovc.ckpt", map_location=torch.device("cpu"))

    G.load_state_dict(g_checkpoint['model'])

    dir = np.random.choice(list_dir, size=1)[0]
    print(dir)
    list_file = get_all_files_in_subdir(dir)

    encoder = VoiceEncoder()

    speakers_embedding = compute_speakers_embedding(files=list_file, encoder=encoder)
    print(speakers_embedding.shape)

    wave_file = np.random.choice(list_file, size=1)[0]

    wav, sample_rate = librosa.load(wave_file, sr=16000)

    duration = wav.shape[0] / sample_rate

    print(duration)

    spectro = preprocess(wav)
    print(spectro.shape)
    window = (spectro.shape[0] / duration) * 0.3
    print(int(window))

    #spectro_input = np.empty(shape=[spectro.shape[0], 128, 80])
#
    #for i in range(spectro.shape[0]):
    #    spectro_slice = spectro[i:i + 128]
    #    pad_len = 128 - spectro_slice.shape[0]
    #    spectro_slice = np.pad(spectro_slice, ((0, pad_len), (0, 0)), mode='constant')
#
    #    print(spectro_slice.shape)
    #    spectro_input[i] = spectro_slice

    pad_len = math.ceil(spectro.shape[0] / 32) * 32 - spectro.shape[0]
    spectro = np.pad(spectro, ((0, pad_len), (0, 0)), mode='constant')

    spectro = torch.FloatTensor(spectro).unsqueeze(0)
    speakers_embedding = torch.FloatTensor(speakers_embedding).unsqueeze(0)

    #spectro = torch.FloatTensor(spectro_input)
    #speakers_embedding = torch.FloatTensor(speakers_embedding).unsqueeze(0)

    content_embedding = G(spectro, speakers_embedding, None)

    print(content_embedding.shape)
