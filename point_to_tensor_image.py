import os

import numpy as np
import cv2

import tensorflow as tf

from line_segments import PARTS, COLORS

import face_alignment
from videostream import QueuedStream


def landmark_points_to_image(landmark_points: np.ndarray, is_2d=True):
    image_list = []
    for landmark in landmark_points:
        image = 255 * np.ones(shape=[1080, 1920, 3], dtype=np.uint8)
        if is_2d:
            landmark = np.reshape(landmark, newshape=[-1, 2])
        else:
            landmark = np.reshape(landmark, newshape=[-1, 3])
        landmark = landmark * np.array([1920, 1080])[np.newaxis, :]
        #print("Landmarks: {}".format(landmark_points))
        for part in PARTS:
            for line in part:
                cv2.line(image, pt1=(int(landmark[line[0]][0]), int(landmark[line[0]][1])),
                         pt2=(int(landmark[line[1]][0]), int(landmark[line[1]][1])), color=COLORS[part],
                         thickness=1)
        image_list.append(image)
    images = np.stack(image_list, axis=0)

    return images


def tf_landmark_points_to_image(landmark_points: tf.placeholder, is_2d: tf.placeholder):
    image = tf.py_func(landmark_points_to_image, [landmark_points, is_2d], tf.uint8)
    return image


video_file = r"C:\Users\Thanh\Downloads\Bill.mp4"
save_dir = r"C:\Users\Thanh\Downloads\landmarks"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

face_al = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cpu", face_detector="sfd")
stream = QueuedStream(uri=video_file)
stream.start()

run = True
sess = tf.Session()
landmark_points = tf.placeholder(dtype=tf.float32, shape=[None, None])
is_2d = tf.placeholder(dtype=tf.bool, shape=[])
image_tf = tf_landmark_points_to_image(landmark_points=landmark_points, is_2d=is_2d)
tf.summary.image("image", image_tf[:, :, :, ::-1])
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("summary")
i = 0


while run:
    ret, frame, frame_id = stream.read()

    if not ret:
        break
    i += 1
    rgb_frame = frame[:, :, ::-1]
    height, width = frame.shape[:2]

    landmarks = face_al.get_landmarks_from_image(rgb_frame)
    print(type(landmarks), len(landmarks))
    landmarks = np.array(landmarks)
    print(landmarks.shape)
    landmarks = landmarks / np.array([width - 1, height - 1])[np.newaxis, np.newaxis, :]
    print(type(landmarks))
    #print("landmarks: {}".format(landmarks))
    landmarks = np.reshape(landmarks, newshape=[-1, landmarks.shape[1] * landmarks.shape[2]])
    #print("landmarks: {}".format(landmarks))

    #image = landmark_points_to_image(landmark_points=landmarks, is_2d=True)

    images = sess.run(image_tf, feed_dict={landmark_points: landmarks, is_2d: True})
    print(images.shape)
    summary = sess.run(merged, feed_dict={landmark_points: landmarks, is_2d: True})
    writer.add_summary(summary=summary, global_step=i)
    #cv2.imshow("Anh", frame)
    j = 0
    for image in images:
        j += 1
        cv2.imshow("Landmark", image)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(save_dir, str(frame_id) + "_" + str(j) + ".jpg"), image)
