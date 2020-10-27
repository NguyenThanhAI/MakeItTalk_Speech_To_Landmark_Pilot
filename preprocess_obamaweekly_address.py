import os
import argparse
from datetime import timedelta

from tqdm import tqdm

import pandas as pd

import cv2

import torch
import face_alignment

from mtcnn import MTCNN

from videostream import QueuedStream


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--videos_dir", type=str, default=r"D:\ObamaWeeklyAddress\videos")
    parser.add_argument("--num_frames_min_of_fragment", type=str, default=2000)
    parser.add_argument("--prob_threshold", type=float, default=0.98)
    parser.add_argument("--area_threshold", type=float, default=0.04)

    args = parser.parse_args()

    return args


def enumerate_video_list(video_dir):
    videos_list = []
    for dirs, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith((".mp4")):
                videos_list.append(os.path.join(dirs, file))
            else:
                os.remove(os.path.join(dirs, file))

    return videos_list


if __name__ == '__main__':
    args = get_args()

    videos_list = enumerate_video_list(args.videos_dir)
    print(len(videos_list))
    detector = MTCNN()
    #detector = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType._2D,
    #                                        device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    info = []
    for video in tqdm(videos_list):
        print(video)
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        stream = QueuedStream(uri=video, fps=int(fps + 10))
        fragments = []
        start = None
        end = None
        stream.start()
        while True:
            ret, frame, frame_id = stream.read()

            if not ret:
                break

            rgb_frame = frame[:, :, ::-1]

            faces = detector.detect_faces(rgb_frame)

            faces = list(filter(lambda x: x["confidence"] > args.prob_threshold and x["box"][2] * x["box"][3] >= args.area_threshold * width * height, faces))

            if len(faces) > 0:
                for face in faces:
                    x, y, w, h = face["box"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    cv2.putText(frame, str(face["confidence"]), (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))
                    cv2.putText(frame, str((w * h) / (width * height)), (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            if len(faces) >= 2:
                print(len(faces))

            if len(faces) == 0 or len(faces) >= 2:
                if start is not None:
                    end = frame_id
                    if end - start >= args.num_frames_min_of_fragment:
                        fragments.append((start, end))
                        start_time = str(timedelta(seconds=start))
                        end_time = str(timedelta(seconds=end))
                    start = None
                    end = None
                else:
                    continue

            elif len(faces) == 1:
                if start is not None:
                    assert end is None
                    continue
                else:
                    start = frame_id

            cv2.destroyAllWindows()
            cv2.imshow(str(frame_id), frame)
            cv2.waitKey(100)

        print(fragments)
        info.append((os.path.basename(video), fragments))

        df = pd.DataFrame(data=info, index=None, columns=["video", "fragments"])
        df.to_csv("fragments.csv", index=False)
