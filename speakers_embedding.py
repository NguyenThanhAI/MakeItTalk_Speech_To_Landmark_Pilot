import os
from tqdm import tqdm
import numpy as np

from resemblyzer import VoiceEncoder, preprocess_wav


def compute_speakers_embedding(files: list or str, encoder: VoiceEncoder):
    if isinstance(files, list):
        embeddings = []
        for file in tqdm(files):
            wav = preprocess_wav(fpath_or_wav=file)
            embedding = encoder.embed_utterance(wav=wav)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        embedding = np.mean(embeddings, axis=0)
    else:
        file = files
        wav = preprocess_wav(fpath_or_wav=file)
        embedding = encoder.embed_utterance(wav=wav)

    return embedding
