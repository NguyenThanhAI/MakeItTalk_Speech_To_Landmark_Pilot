import os
import subprocess
import argparse

from datetime import timedelta
import time

from tqdm import tqdm

import pandas as pd

import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_file", type=str, default="fragments.csv")
    parser.add_argument("--videos_dir", type=str, default=r"D:\ObamaWeeklyAddress\videos")
    parser.add_argument("--videos_saved_dir", type=str, default=r"D:\ObamaWeeklyAddress\cutted_videos")
    parser.add_argument("--audios_saved_dir", type=str, default=r"D:\ObamaWeeklyAddress\cutted_audios")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.videos_saved_dir):
        os.makedirs(args.videos_saved_dir)

    if not os.path.exists(args.audios_saved_dir):
        os.makedirs(args.audios_saved_dir)

    df = pd.read_csv(filepath_or_buffer=args.csv_file)

    print(df)

    i = 0

    for index, row in tqdm(df.iterrows()):
        video_name = row["video"]
        frame_fragments = eval(row["fragments"])
        if len(frame_fragments) == 0:
            continue

        cap = cv2.VideoCapture(os.path.join(args.videos_dir, video_name))
        fps = cap.get(cv2.CAP_PROP_FPS)

        for fragment in frame_fragments:
            #print(video_name, fragment)
            start_frame = fragment[0] + int(fps)
            end_frame = fragment[1] - int(fps)

            start_ts = str(timedelta(seconds=int(start_frame / fps)))
            end_ts = str(timedelta(seconds=int(end_frame / fps)))

            out_video_file = os.path.join(args.videos_saved_dir, str(i) + ".mp4")
            out_audio_file = os.path.join(args.audios_saved_dir, str(i) + ".wav")

            #print(start_ts, end_ts)

            #print("ffmpeg -i {} -ss {} -to {} {}".format(video_name, start_ts, end_ts, out_video_file))
            #print("ffmpeg -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}".format(out_video_file, out_audio_file))

            subprocess.run(["ffmpeg", "-i", os.path.join(args.videos_dir, video_name), "-ss", start_ts, "-to", end_ts, out_video_file])
            subprocess.run(["ffmpeg", "-i", out_video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", out_audio_file])

            i += 1
