import os

import numpy as np
import tensorflow as tf

from ops import multi_stack_lstm


context_features = {"length": tf.FixedLenFeature(shape=[], dtype=tf.int64),
                    "static_landmark": tf.FixedLenFeature(shape=[204], dtype=tf.float32),
                    "speaker_embedding": tf.FixedLenFeature(shape=[256], dtype=tf.float32)}

sequence_features = {"content_embedding": tf.FixedLenSequenceFeature([64], dtype=tf.float32),
                     "landmark_label": tf.FixedLenSequenceFeature([204], dtype=tf.float32)}



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
    landmark_label.set_shape([None, 204])

    return length, static_landmark, speaker_embedding, content_embedding, landmark_label


def enumerate_tfrecord(dir):
    tfrecord_list = []
    for dirs, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".tfrecord"):
                tfrecord_list.append(os.path.join(dirs, file))

    return tfrecord_list


def get_batch(tfrecord_path, batch_size, num_epochs=100):
    if os.path.isdir(tfrecord_path):
        tfrecord_list = enumerate_tfrecord(tfrecord_path)
        dataset = tf.data.TFRecordDataset(tfrecord_list)
    else:
        dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_single_sequence_example)
    dataset = dataset.shuffle(2000)
    epoch = tf.data.Dataset.range(num_epochs)
    dataset = epoch.flat_map(lambda i: tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensors(i).repeat())))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=([], [204], [256], [None, 64], [None, 204]))

    iterator = dataset.make_one_shot_iterator()

    (length, static_landmark, speaker_embedding, content_embedding, landmark_label), epoch_now = iterator.get_next()

    return length, static_landmark, speaker_embedding, content_embedding, landmark_label, epoch_now


batch_size = 5
time_steps = 100
hidden_size = 256

dataset = tf.data.TFRecordDataset(["sequence_example.tfrecord"])
dataset = dataset.map(_parse_single_sequence_example)
dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=([], [136], [256], [None, 64], [None, 136]))
#dataset = dataset.batch(1)
dataset = dataset.shuffle(100)
dataset = dataset.repeat(1)

iterator = dataset.make_one_shot_iterator()
length, static_landmark, speaker_embedding, content_embedding, landmark_label = iterator.get_next()
print(length, static_landmark, speaker_embedding, content_embedding, landmark_label)
#content_embedding.set_shape([None, None, 64])
content_embedding = tf.pad(content_embedding, paddings=[(0, 0), (0, 18), (0, 0)])
stacked_lstm = multi_stack_lstm(hidden_size=hidden_size, num_lstms=3)
initial_state = state = stacked_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

print(initial_state)

outputs, states = tf.nn.dynamic_rnn(stacked_lstm, content_embedding, initial_state=initial_state, time_major=False, sequence_length=length)
outputs = outputs[:, 18:, :]

mask = tf.sequence_mask(lengths=length, maxlen=tf.shape(outputs)[1])

print(outputs, states)

initializer = tf.global_variables_initializer()


i = 0
with tf.Session() as sess:
    sess.run(initializer)
    try:
        while True:
            i += 1
            m, out, l, sl, se, ce, ll = sess.run([mask, outputs, length, static_landmark, speaker_embedding, content_embedding, landmark_label])
            print(i, l, sl.shape, se.shape, ce.shape, ll.shape, m.shape)
            print(out.shape)
    except:
        print("Training process finished")
        pass
