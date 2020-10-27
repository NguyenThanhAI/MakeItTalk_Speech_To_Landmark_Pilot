import time

import numpy as np
import cv2

import torch
import face_alignment

from icp import icp

from line_segments import PARTS, COLORS

device = 'cuda:0' if torch.cuda.is_available() else "cpu"
facial_landmark_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)

#standard_landmark = np.load("mean_shape.npy")

obama_image = cv2.imread(r"C:\Users\Thanh\Downloads\Official_portrait_of_Barack_Obama.jpg")

landmark = facial_landmark_detector.get_landmarks_from_image(obama_image[:, :, ::-1])
assert isinstance(landmark, list)
assert len(landmark) == 1
landmark = landmark[0]

standard_landmark = landmark
#
#image = 255 * np.ones_like(obama_image)
#
#for part in PARTS:
#    for line in part:
#        cv2.line(image, pt1=(int(landmark[line[0]][0]), int(landmark[line[0]][1])),
#                 pt2=(int(landmark[line[1]][0]), int(landmark[line[1]][1])), color=COLORS[part],
#                 thickness=1)
#
#for i, point in enumerate(landmark):
#    x, y = point
#    cv2.putText(image, str(i), (int(x + 5), int(y + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
#
#cv2.imshow("Anh", image)
#cv2.waitKey(0)

# 36 - (0.35, 0.3) 45 - (0.65, 0.3) 33 - (0.5, 0.6)

#print(standard_landmark)

max_x, max_y = np.max(standard_landmark, axis=0)

print(max_x, max_y)

#image = 255 * np.ones(shape=[310, 310, 3], dtype=np.uint8)
image = 255 * np.ones_like(obama_image)

for part in PARTS:
    for line in part:
        cv2.line(image, pt1=(int(standard_landmark[line[0]][0]), int(standard_landmark[line[0]][1])),
                 pt2=(int(standard_landmark[line[1]][0]), int(standard_landmark[line[1]][1])), color=COLORS[part],
                 thickness=1)

cv2.circle(image, (int(standard_landmark[36][0]), int(standard_landmark[36][1])), 2, (0, 0, 255), thickness=-1)
cv2.circle(image, (int(standard_landmark[45][0]), int(standard_landmark[45][1])), 2, (0, 0, 255), thickness=-1)
cv2.circle(image, (int(standard_landmark[48][0]), int(standard_landmark[48][1])), 2, (0, 0, 255), thickness=-1)
cv2.circle(image, (int(standard_landmark[54][0]), int(standard_landmark[54][1])), 2, (0, 0, 255), thickness=-1)
cv2.circle(image, (int(standard_landmark[8][0]), int(standard_landmark[8][1])), 2, (0, 0, 255), thickness=-1)

cv2.imwrite("Anh.jpg", image)

width, height = 480, 600

aligned_image = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)
#
#target_eye_corner = np.float32([[0.3 * width, height / 3], [0.7 * width, height / 3], [0.4 * width, height / 1.5], [0.6 * width, height / 1.5]])
#print(target_eye_corner)
#source_eye_corner = np.float32([[standard_landmark[36, 0], standard_landmark[36, 1]], [standard_landmark[45, 0], standard_landmark[45, 1]],
#                                [standard_landmark[48, 0], standard_landmark[48, 1]], [standard_landmark[54, 0], standard_landmark[54, 1]]])
#print(source_eye_corner)

target_eye_corner = np.float32([[0.25 * width, 0.25 * height], [0.75 * width, 0.25 * height], [0.5 * width, 0.5 * height]])
print(target_eye_corner)
source_eye_corner = np.float32([[standard_landmark[36, 0], standard_landmark[36, 1]], [standard_landmark[45, 0], standard_landmark[45, 1]],
                                [standard_landmark[33, 0], standard_landmark[33, 1]]])
#print(source_eye_corner)
#pts1 = np.float32([[0, 260], [640, 260], [0, 400], [640, 400]])
#pts2 = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])

#h = cv2.getPerspectiveTransform(source_eye_corner, target_eye_corner)

start = time.time()
h, status = cv2.estimateAffine2D(source_eye_corner, target_eye_corner)
end = time.time()

#h, status = cv2.findHomography(source_eye_corner, target_eye_corner)

print(h, end - start)

point_out = cv2.transform(np.expand_dims(source_eye_corner, axis=1), h)

point_out = np.squeeze(point_out)

print(point_out)

transformed_landmark = cv2.transform(np.expand_dims(standard_landmark, axis=1), h)

transformed_landmark = np.squeeze(transformed_landmark)

print(transformed_landmark)

for part in PARTS:
    for line in part:
        cv2.line(aligned_image, pt1=(int(transformed_landmark[line[0]][0]), int(transformed_landmark[line[0]][1])),
                 pt2=(int(transformed_landmark[line[1]][0]), int(transformed_landmark[line[1]][1])), color=COLORS[part],
                 thickness=1)

cv2.imwrite("Aligned_Anh.jpg", aligned_image)
