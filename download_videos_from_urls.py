import os
import argparse

from tqdm import tqdm

import urllib3
import requests
import asyncio

from urllib.error import HTTPError

import youtube_dl


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url_text_path", type=str, default=r"D:\ObamaWeeklyAddress\VideosLinkList.txt")
    parser.add_argument("--save_video_dir", type=str, default=r"D:\ObamaWeeklyAddress\videos")
    parser.add_argument("--save_wav_dir", type=str, default=r"D:\ObamaWeeklyAddress\wavs")

    args = parser.parse_args()

    return args


def read_text_file(file_path: str):
    with open(file_path, "r") as f:
        lines = f.read()
        lines = lines.split("\n")
        lines = list(filter(None, lines))
        f.close()
    return lines


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.save_video_dir):
        os.makedirs(args.save_video_dir, exist_ok=True)
    if not os.path.exists(args.save_wav_dir):
        os.makedirs(args.save_wav_dir, exist_ok=True)

    urls = read_text_file(args.url_text_path)

    print("urls: {}".format(urls))
    ydl_opts = {}
    os.chdir(args.save_video_dir)
    for url in tqdm(urls):
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except youtube_dl.utils.DownloadError:
            print("Error and continue")
            continue
