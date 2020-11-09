import os
import argparse

import numpy as np
import cv2

import tensorflow as tf

from line_segments import PARTS, COLORS


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecord_dir", type=str, default=r"D:\ObamaWeeklyAddress\tfrecord")

    args = parser.parse_args()

    return args


def enumerate_tfrecord(dir):
    tfrecord_list = []
    for dirs, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".tfrecord"):
                tfrecord_list.append(os.path.join(dirs, file))

    return tfrecord_list


context_features = {"length": tf.FixedLenFeature(shape=[], dtype=tf.int64),
                    "static_landmark": tf.FixedLenFeature(shape=[136], dtype=tf.float32),
                    "speaker_embedding": tf.FixedLenFeature(shape=[256], dtype=tf.float32)}

sequence_features = {"content_embedding": tf.FixedLenSequenceFeature([64], dtype=tf.float32),
                     "landmark_label": tf.FixedLenSequenceFeature([136], dtype=tf.float32)}


def _parse_single_sequence_example(data_record):
    context_sample, sequence_sample = tf.parse_single_sequence_example(serialized=data_record,
                                                                       context_features=context_features,
                                                                       sequence_features=sequence_features)
    length = context_sample["length"]
    static_landmark = context_sample["static_landmark"]
    speaker_embedding = context_sample["speaker_embedding"]
    content_embedding = sequence_sample["content_embedding"]
    landmark_label = sequence_sample["landmark_label"]

    content_embedding.set_shape([None, 64])
    landmark_label.set_shape([None, 136])

    return length, static_landmark, speaker_embedding, content_embedding, landmark_label


def get_batch(tfrecord_path, batch_size=1, num_epochs=1):
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    dataset = dataset.map(_parse_single_sequence_example)
    dataset = dataset.repeat(num_epochs)
    #dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=([], [136], [256], [None, 64], [None, 136]))
    dataset = dataset.batch(batch_size=batch_size)

    iterator = dataset.make_one_shot_iterator()

    length, static_landmark, speaker_embedding, content_embedding, landmark_label = iterator.get_next()

    return length, static_landmark, speaker_embedding, content_embedding, landmark_label


if __name__ == '__main__':
    args = get_args()

    tfrecord_list = enumerate_tfrecord(args.tfrecord_dir)

    for tfrecord_path in tfrecord_list:
        length, static_landmark, speaker_embedding, content_embedding, landmark_label = get_batch(tfrecord_path=tfrecord_path)

        with tf.Session() as sess:
            try:
                while True:
                    l, sl, se, ce, ll = sess.run([length, static_landmark, speaker_embedding, content_embedding, landmark_label])
                    print(l, sl.shape, se.shape, ce.shape, ll.shape)
                    sl = np.squeeze(sl)
                    ll = np.squeeze(ll)
                    if np.all(np.isnan(ll)):
                        print(tfrecord_path)
                        sess.close()
                        break
                    try:
                        for landmark in ll:
                            frame = 255 * np.ones(shape=[600, 480, 3], dtype=np.uint8)
                            landmark = np.reshape(landmark, newshape=[-1, 2])
                            for part in PARTS:
                                for line in part:
                                    cv2.line(frame, pt1=(int(landmark[line[0]][0]), int(landmark[line[0]][1])),
                                             pt2=(int(landmark[line[1]][0]), int(landmark[line[1]][1])), color=COLORS[part],
                                             thickness=1)
                            cv2.imshow("landmark label", frame)
                            cv2.waitKey(15)
                    except:
                        cv2.destroyAllWindows()
                        sess.close()
                        print(tfrecord_path)
                        break
                    cv2.destroyAllWindows()

            except tf.errors.OutOfRangeError:
                sess.close()
                pass