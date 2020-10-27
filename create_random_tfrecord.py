from tqdm import tqdm

import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    if  not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):

    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _image_to_tfexample(image_data, label, width, height, channels):
    return tf.train.Example(features=tf.train.Features(feature={'image': _bytes_feature(image_data),
                                                                'label': _int64_feature(label),
                                                                'width': _int64_feature(width),
                                                                'height': _int64_feature(height),
                                                                'channels': _int64_feature(channels)}))


def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) for value in values])


def _floats_feature_list(values):
    return tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=list(value))) for value in values])


def _sequence_example(static_landmark: np.ndarray, speaker_embed: np.ndarray, audio_frames: np.ndarray, landmark_frames: np.ndarray) -> tf.train.SequenceExample:
    length = audio_frames.shape[0]
    #audio_frames = audio_frames.tolist()
    #landmark_frames = landmark_frames.tolist()
    #print(_floats_feature_list(audio_frames), _floats_feature_list(landmark_frames))
    #print([tf.train.Feature(float_list=tf.train.FloatList(value=list(value))) for value in audio_frames],
    #      [tf.train.Feature(float_list=tf.train.FloatList(value=list(value))) for value in landmark_frames])
    sequence_example = tf.train.SequenceExample(context=tf.train.Features(feature={"length": _int64_feature(length),
                                                                                   "static_landmark": _float_feature(static_landmark),
                                                                                   "speaker_embedding": _float_feature(speaker_embed)}),
                                                feature_lists=tf.train.FeatureLists(feature_list={"content_embedding": _floats_feature_list(audio_frames),
                                                                                                  "landmark_label": _floats_feature_list(landmark_frames)}))
    #sequence_example = tf.train.Example(features=tf.train.Features(feature={"length": _int64_feature(length),
    #                                                                        "speaker_embedding": _float_feature(speaker_embed)}))
    #sequence_example = tf.train.SequenceExample(context=tf.train.Features(feature={"length": _int64_feature(length),
    #                                                                               "speaker_embedding": _float_feature(speaker_embed)}),
    #                                            feature_lists=tf.train.FeatureLists(feature_list={"content_embedding": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=list(value))) for value in audio_frames]),
    #                                                                                              "landmark_label": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=list(value))) for value in landmark_frames])}))
    return sequence_example

static_landmark_size = 204
speaker_embedding_size = 256
embedding_size = 64
label_landmark_size = 204

num_examples = 1000

with tf.python_io.TFRecordWriter("sequence_example.tfrecord") as tfrecord_writer:
    for _ in tqdm(range(num_examples)):
        length = np.random.randint(10, 50, size=1)[0]
        static_landmark = np.random.rand(static_landmark_size)
        speaker_embedding = np.random.rand(speaker_embedding_size)
        content_embedding = np.random.rand(length, embedding_size)
        label_landmark = np.random.rand(length, label_landmark_size)
        #print(content_embedding.shape)
        #for value in content_embedding:
        #    print(value.shape)
        sequence_example = _sequence_example(static_landmark=static_landmark,
                                             speaker_embed=speaker_embedding,
                                             audio_frames=content_embedding,
                                             landmark_frames=label_landmark)

        tfrecord_writer.write(sequence_example.SerializeToString())

    tfrecord_writer.close()
