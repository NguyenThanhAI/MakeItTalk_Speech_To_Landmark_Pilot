import os

import tensorflow as tf


def enumerate_tfrecord(dir):
    tfrecord_list = []
    for dirs, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".tfrecord"):
                tfrecord_list.append(os.path.join(dirs, file))

    return tfrecord_list


def get_context_sequence_features(is_2d=True):
    if is_2d:
        num_points = 136
    else:
        num_points = 204
    context_features = {"length": tf.FixedLenFeature(shape=[], dtype=tf.int64),
                        "static_landmark": tf.FixedLenFeature(shape=[num_points], dtype=tf.float32),
                        "speaker_embedding": tf.FixedLenFeature(shape=[256], dtype=tf.float32)}

    sequence_features = {"content_embedding": tf.FixedLenSequenceFeature([64], dtype=tf.float32),
                         "landmark_label": tf.FixedLenSequenceFeature([num_points], dtype=tf.float32)}

    return context_features, sequence_features


context_features, sequence_features = get_context_sequence_features(is_2d=True)


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


def get_batch(tfrecord_path, batch_size=2, num_epochs=100):
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
    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(([], [136], [256], [None, 64], [None, 136]), []))

    iterator = dataset.make_one_shot_iterator()

    (length, static_landmark, speaker_embedding, content_embedding, landmark_label), epoch_now = iterator.get_next()
    return length, static_landmark, speaker_embedding, content_embedding, landmark_label, epoch_now
