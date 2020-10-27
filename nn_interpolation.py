import numpy as np

time_length = 6.2

content_embedding = np.random.rand(2000, 64)
landmarks = np.random.rand(500, 136)

content_dt = time_length / (content_embedding.shape[0] - 1)
landmarks_dt = time_length / (landmarks.shape[0] - 1)
print(content_dt, landmarks_dt)

content_time_step = np.linspace(0, stop=time_length, num=content_embedding.shape[0], endpoint=True)
landmarks_time_step = np.linspace(0, stop=time_length, num=landmarks.shape[0], endpoint=True)

print("content_time_step: {} \n landmarks_time_step: {}".format(content_time_step, landmarks_time_step))

from scipy.interpolate import NearestNDInterpolator
from sklearn.neighbors import KDTree

tree = KDTree(np.expand_dims(landmarks_time_step, axis=1))

index = tree.query(np.expand_dims(content_time_step, axis=1), return_distance=False)

respective_landmark = landmarks[np.squeeze(index)]
print(respective_landmark.shape)
