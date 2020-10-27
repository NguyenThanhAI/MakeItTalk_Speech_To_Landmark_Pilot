import os
import argparse
import subprocess

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", type=str, default=r"D:\ObamaWeeklyAddress\videos")

    args = parser.parse_args()

    return args


def enumerate_video_list(video_dir):
    videos_list = []
    for dirs, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith((".webm", ".mkv")):
                videos_list.append(os.path.join(dirs, file))

    return videos_list


if __name__ == '__main__':

    args = get_args()

    videos_list = enumerate_video_list(video_dir=args.video_dir)

    print(videos_list)

    for video in tqdm(videos_list):
        out_file = os.path.splitext(video)[0] + ".mp4"
        subprocess.run(["ffmpeg", "-i", video, out_file])
