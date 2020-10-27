import tensorflow as tf


#parts = [tf.range(17), tf.range(48, 68), tf.range(27, 36), tf.range(17, 22), tf.range(22, 27), tf.range(36, 42), tf.range(42, 48)]
#parts = [(0, 17), (48, 68), (27, 36), (17, 22), (22, 27), (36, 42), (42, 48)]
parts = [(0, 17), (17, 22), (22, 27), (27, 36), (36, 42), (42, 48), (48, 68)]

def l2_loss(predict: tf.Tensor, groundtruth: tf.Tensor, sequence_mask: tf.Tensor):

    error = tf.square(predict - groundtruth)
    error = tf.reduce_sum(error, axis=2)
    sequence_mask = tf.cast(sequence_mask, dtype=tf.float32)
    error = tf.multiply(sequence_mask, error)
    error = tf.reduce_sum(error, axis=1)
    error /= tf.reduce_sum(sequence_mask, axis=1)
    error = tf.reduce_mean(error)

    return error


def graph_laplacian(predict: tf.Tensor, groundtruth: tf.Tensor, sequence_mask: tf.Tensor, is_2d=True):
    if is_2d:
        predict_shape = tf.shape(predict)
        predict = tf.reshape(predict, shape=[predict_shape[0], predict_shape[1], -1, 2])
        groundtruth_shape = tf.shape(groundtruth)
        groundtruth = tf.reshape(groundtruth, shape=[groundtruth_shape[0], groundtruth_shape[1], -1, 2])
    else:
        predict_shape = tf.shape(predict)
        predict = tf.reshape(predict, shape=[predict_shape[0], predict_shape[1], -1, 3])
        groundtruth_shape = tf.shape(groundtruth)
        groundtruth = tf.reshape(groundtruth, shape=[groundtruth_shape[0], groundtruth_shape[1], -1, 3])

    # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    # [1, 2, 5]
    # [0., 2.67, 2.67, 3., 4., 2.67, 6., 7., 8., 9.]
    #predict_copy = tf.identity(predict)
    #groundtruth_copy = tf.identity(groundtruth)
    predict_copy = []
    groundtruth_copy = []
    for part in parts:
        predict_mean = tf.tile(tf.reduce_mean(predict[:, :, part[0]:part[1], :], axis=2, keepdims=True), multiples=[1, 1, part[1] - part[0], 1])
        #predict_copy[:, :, part[0]:part[1], :] = tf.identity(predict_mean) # TensorFlow does not support items assignment
        predict_copy.append(predict_mean)
        groundtruth_mean = tf.tile(tf.reduce_mean(groundtruth[:, :, part[0]:part[1], :], axis=2, keepdims=True), multiples=[1, 1, part[1] - part[0], 1])
        #groundtruth_copy[:, :, part[0]:part[1], :] = tf.identity(groundtruth_mean) # TensorFlow does not support items assignment
        groundtruth_copy.append(groundtruth_mean)
    predict_copy = tf.concat(predict_copy, axis=2)
    groundtruth_copy = tf.concat(groundtruth_copy, axis=2)
    error_predict = predict - predict_copy
    error_groundtruth = groundtruth - groundtruth_copy
    error = tf.square(error_predict - error_groundtruth)
    error = tf.reduce_sum(error, axis=[2, 3])
    sequence_mask = tf.cast(sequence_mask, dtype=tf.float32)
    error = tf.multiply(sequence_mask, error)
    error = tf.reduce_sum(error, axis=1)
    error /= tf.reduce_sum(sequence_mask, axis=1)
    error = tf.reduce_mean(error)

    return error


def discriminator_loss(real_scores: tf.Tensor, synthesized_scores: tf.Tensor, sequence_mask:tf.Tensor):
    sequence_mask = tf.cast(sequence_mask, dtype=tf.float32)

    real_scores = tf.squeeze(real_scores, axis=2)
    synthesized_scores = tf.squeeze(synthesized_scores, axis=2)

    real_loss = tf.square((real_scores - tf.ones_like(real_scores)))
    real_loss = tf.multiply(sequence_mask, real_loss)
    real_loss = tf.reduce_sum(real_loss, axis=1)
    real_loss = tf.reduce_mean(real_loss)

    fake_loss = tf.square(synthesized_scores)
    fake_loss = tf.multiply(sequence_mask, fake_loss)
    fake_loss = tf.reduce_sum(fake_loss, axis=1)
    fake_loss = tf.reduce_mean(fake_loss)

    discriminator_loss = real_loss + fake_loss

    return discriminator_loss


def adversarial_loss(synthesized_scores: tf.Tensor, sequence_mask: tf.Tensor):
    sequence_mask = tf.cast(sequence_mask, dtype=tf.float32)

    synthesized_scores = tf.squeeze(synthesized_scores, axis=2)

    loss = tf.square(synthesized_scores - tf.ones_like(synthesized_scores))
    loss = tf.multiply(sequence_mask, loss)
    loss = tf.reduce_sum(loss, axis=1)
    loss = tf.reduce_mean(loss)

    return loss


#batch_size = 5
#max_length = 100
#landmark_size = 204
#
#context_features = {"length": tf.FixedLenFeature(shape=[], dtype=tf.int64),
#                    "speaker_embedding": tf.FixedLenFeature(shape=[256], dtype=tf.float32)}
#
#sequence_features = {"content_embedding": tf.FixedLenSequenceFeature([64], dtype=tf.float32),
#                     "landmark_label": tf.FixedLenSequenceFeature([204], dtype=tf.float32)}
#
#
#
#def _parse_single_sequence_example(data_record):
#    context_sample, sequence_sample = tf.parse_single_sequence_example(serialized=data_record,
#                                                                       context_features=context_features,
#                                                                       sequence_features=sequence_features)
#    length = context_sample["length"]
#    speaker_embedding = context_sample["speaker_embedding"]
#    content_embedding = sequence_sample["content_embedding"]
#    landmark_label = sequence_sample["landmark_label"]
#
#    content_embedding.set_shape([None, 64])
#    landmark_label.set_shape([None, 204])
#
#    return length, speaker_embedding, content_embedding, landmark_label
#
#dataset = tf.data.TFRecordDataset(["sequence_example.tfrecord"])
#dataset = dataset.map(_parse_single_sequence_example)
#dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=([], [256], [None, 64], [None, 204]))
##dataset = dataset.batch(1)
#dataset = dataset.shuffle(100)
#dataset = dataset.repeat(1)
#
#iterator = dataset.make_one_shot_iterator()
#length, speaker_embedding, content_embedding, landmark_label = iterator.get_next()
#
#predict = tf.random.uniform(shape=[tf.shape(content_embedding)[0], tf.shape(content_embedding)[1], 204], maxval=1., dtype=tf.float32)
#groundtruth = tf.random.uniform(shape=[tf.shape(content_embedding)[0], tf.shape(content_embedding)[1], 204], maxval=1., dtype=tf.float32)
#sequence_mask = tf.sequence_mask(lengths=length, maxlen=tf.reduce_max(length))
##loss = l2_loss(predict=predict, groundtruth=groundtruth, sequence_mask=sequence_mask)
#loss = graph_laplacian(predict=predict, groundtruth=groundtruth, sequence_mask=sequence_mask, is_2d=False)
#
#with tf.Session() as sess:
#    for i in range(100):
#        cost = sess.run(loss)
#        print(cost)
