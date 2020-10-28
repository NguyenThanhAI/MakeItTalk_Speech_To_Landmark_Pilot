import os
import argparse

import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", type=str, default=r"D:\ObamaWeeklyAddress\cutted_videos")

    args = parser.parse_args()

    return args


def enumerate_videos(dir):
    videos_list = []
    for dirs, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".mp4"):
                videos_list.append(os.path.join(dirs, file))

    return videos_list

if __name__ == '__main__':
    args = get_args()

    videos_list = enumerate_videos(args.video_dir)

    for video in videos_list:
        cap = cv2.VideoCapture(video)
        print("video: {}, cap: {}".format(video, cap.isOpened()))
        cap.release()
