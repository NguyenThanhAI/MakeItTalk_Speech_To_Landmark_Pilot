import os

import numpy as np
import cv2

import torch

import face_alignment
from videostream import QueuedStream

from line_segments import PARTS, COLORS

video_path = r"D:\ObamaWeeklyAddress\videos\20170107 Weekly Address HD-7G5kMmnAp_8.mp4"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
landmark_detector = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType._2D, device=device)

stream = QueuedStream(video_path)
stream.start()
landmark_list = []
while True:
    ret, frame, frame_id = stream.read()
    if not ret:
        break
    rgb_frame = frame[:, :, ::-1]
    landmarks = landmark_detector.get_landmarks_from_image(rgb_frame)
    if landmarks is None:
        continue
    else:
        landmark = max(landmarks, key=lambda x: np.prod(np.max(x, axis=0) - np.min(x, axis=0)))
        for part in PARTS:
            for line in part:
                cv2.line(frame, pt1=(int(landmark[line[0]][0]), int(landmark[line[0]][1])),
                         pt2=(int(landmark[line[1]][0]), int(landmark[line[1]][1])), color=COLORS[part],
                         thickness=1)
        landmark_list.append(landmark)
    cv2.imshow("Anh", frame)
    cv2.waitKey(1000)

landmark_list = np.stack(landmark_list, axis=0)
