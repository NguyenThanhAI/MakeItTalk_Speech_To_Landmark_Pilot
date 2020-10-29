import os
import argparse

import math
import random

from tqdm import tqdm

import librosa
import numpy as np

import cv2

import tensorflow as tf

import torch
import torchvision

import face_alignment

from sklearn.neighbors import KDTree

from resemblyzer import VoiceEncoder, preprocess_wav

from speakers_embedding import compute_speakers_embedding
from preprocess import preprocess

from model_vc import Generator, Encoder


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio_dir", type=str, default=r"D:\ObamaWeeklyAddress\cutted_audios")
    parser.add_argument("--video_dir", type=str, default=r"D:\ObamaWeeklyAddress\cutted_videos")
    parser.add_argument("--out_dir", type=str, default=r"D:\ObamaWeeklyAddress\tfrecord")
    parser.add_argument("--auto_vc_checkpoint", type=str, default=r"E:\PythonProjects\Face_Animation\autovc\autovc.ckpt")
    parser.add_argument("--static_image", type=str, default=r"C:\Users\Thanh\Downloads\Official_portrait_of_Barack_Obama.jpg")
    parser.add_argument("--index_file", type=str, default="obamaweeklyaddress_index.txt")
    parser.add_argument("--target_width", type=int, default=480)
    parser.add_argument("--target_height", type=int, default=600)

    args = parser.parse_args()
    return args


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):

    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) for value in values])


def _floats_feature_list(values):
    return tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=list(value))) for value in values])


def _sequence_example(static_landmark: np.ndarray, speaker_embed: np.ndarray, audio_frames: np.ndarray, landmark_frames: np.ndarray) -> tf.train.SequenceExample:
    length = audio_frames.shape[0]
    sequence_example = tf.train.SequenceExample(context=tf.train.Features(feature={"length": _int64_feature(length),
                                                                                   "static_landmark": _float_feature(static_landmark),
                                                                                   "speaker_embedding": _float_feature(speaker_embed)}),
                                                feature_lists=tf.train.FeatureLists(feature_list={"content_embedding": _floats_feature_list(audio_frames),
                                                                                                  "landmark_label": _floats_feature_list(landmark_frames)}))
    return sequence_example


def enumerate_audio_video_file(dir):
    audio_file_list = []
    for dirs, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".mp4"):
                audio_file_list.append(os.path.join(dirs, file))

    return audio_file_list


def read_wav_file(wav_path):
    wav, sample_rate = librosa.load(wav_path, sr=16000)
    duration = wav.shape[0] / sample_rate

    return wav, duration


def pair_audio_and_video_file(audio_dir, video_dir):
    video_file_list = enumerate_audio_video_file(video_dir)
    audios_to_videos = dict(map(lambda x: (os.path.join(audio_dir, os.path.splitext(os.path.relpath(x, video_dir))[0] + ".wav"), x), video_file_list))
    return audios_to_videos


def align_landmark(landmark: np.ndarray, width: int, height: int):
    target_eye_corner = np.float32(
        [[0.25 * width, 0.25 * height], [0.75 * width, 0.25 * height], [0.5 * width, 0.5 * height]])

    source_eye_corner = np.float32([[landmark[36, 0], landmark[36, 1]],
                                    [landmark[45, 0], landmark[45, 1]],
                                    [landmark[33, 0], landmark[33, 1]]])

    h, status = cv2.estimateAffine2D(source_eye_corner, target_eye_corner)

    landmark = cv2.transform(np.expand_dims(landmark, axis=1), h)
    landmark = np.squeeze(landmark)
    landmark = np.reshape(landmark, newshape=[-1])

    return landmark


def extract_landmarks_from_videos(video_path: str, landmark_detector: face_alignment.FaceAlignment, width=480, height=600):
    cap = cv2.VideoCapture(video_path)
    landmark_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]
        landmarks = landmark_detector.get_landmarks_from_image(rgb_frame)

        if landmarks is None:
            continue
        else:
            landmark = max(landmarks, key=lambda x: np.prod(np.max(x, axis=0) - np.min(x, axis=0)))

            landmark = align_landmark(landmark=landmark, width=width, height=height)
            landmark_list.append(landmark)

    landmark_list = np.stack(landmark_list, axis=0)

    return landmark_list


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    G = Generator(32, 256, 512, 32).eval().to(device)

    if torch.cuda.is_available():
        g_checkpoint = torch.load(args.auto_vc_checkpoint, map_location=torch.device(device))
    else:
        g_checkpoint = torch.load(args.auto_vc_checkpoint, map_location=torch.device("cpu"))

    G.load_state_dict(g_checkpoint['model'])

    landmark_detector = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType._2D,
                                                     device=device)

    static_image = cv2.imread(args.static_image)
    static_image = static_image[:, :, ::-1]

    standard_landmark = landmark_detector.get_landmarks_from_image(static_image)
    assert len(standard_landmark) == 1
    standard_landmark = standard_landmark[0]
    static_landmark = align_landmark(landmark=standard_landmark, width=args.target_width,
                                     height=args.target_height)

    audios_to_videos = pair_audio_and_video_file(audio_dir=args.audio_dir, video_dir=args.video_dir)

    encoder = VoiceEncoder()

    if os.path.exists(args.index_file):
        with open(args.index_file, "r") as f:
            index_file = int(f.read())
            f.close()
    else:
        index_file = 0

    audio_list = list(audios_to_videos.keys())
    speaker_embedding = compute_speakers_embedding(files=audio_list, encoder=encoder)
    print("index file: {}".format(index_file))
    for audio in tqdm(audio_list[index_file:]):
        with tf.python_io.TFRecordWriter(os.path.join(args.out_dir, "obamaweeklyaddress_training_" + str(index_file) + ".tfrecord")) as tfrecord_writer:
            video = audios_to_videos[audio]
            landmarks = extract_landmarks_from_videos(video_path=video, landmark_detector=landmark_detector,
                                                      width=args.target_width, height=args.target_height)
            wav, duration = read_wav_file(wav_path=audio)

            spectro = preprocess(wav)
            pad_len = math.ceil(spectro.shape[0] / 32) * 32 - spectro.shape[0]
            spectro = np.pad(spectro, ((0, pad_len), (0, 0)), mode="constant")

            spectro_pt = torch.FloatTensor(spectro).unsqueeze(0)
            speaker_embedding_pt = torch.FloatTensor(speaker_embedding).unsqueeze(0)

            if torch.cuda.is_available():
                spectro_pt = spectro_pt.to(device)
                speaker_embedding_pt = speaker_embedding_pt.to(device)

            content_embedding_pt = G(spectro_pt, speaker_embedding_pt, None)
            if torch.cuda.is_available():
                content_embedding = content_embedding_pt.cpu().detach().numpy()
            else:
                content_embedding = content_embedding_pt.detach().numpy()
            content_embedding = content_embedding[0, :-pad_len, :]

            content_time_step = np.linspace(0, stop=duration, num=content_embedding.shape[0], endpoint=True)
            landmarks_time_step = np.linspace(0, stop=duration, num=landmarks.shape[0], endpoint=True)

            tree = KDTree(np.expand_dims(landmarks_time_step, axis=1))

            index = tree.query(np.expand_dims(content_time_step, axis=1), return_distance=False)

            respective_landmarks = landmarks[np.squeeze(index)]

            assert content_embedding.shape[0] == respective_landmarks.shape[0]

            sequence_example = _sequence_example(static_landmark=static_landmark,
                                                 speaker_embed=speaker_embedding,
                                                 audio_frames=content_embedding,
                                                 landmark_frames=respective_landmarks)

            tfrecord_writer.write(sequence_example.SerializeToString())
            tfrecord_writer.close()

        index_file += 1
        with open(args.index_file, "w") as f:
            f.write(str(index_file))
            f.close()
