import numpy as np

#import librosa
#import soundfile
#
#from dsp import load_wav, save_wav, linear_to_mel, build_mel_basis, normalize, amp_to_db, melspectrogram, stft
#
#wave_file_path = r"D:\VCTK_Corpus\VCTK-Corpus\wav48\p225\p225_001.wav"
#
#wav_1 = load_wav(wave_file_path)
#
#wav_2, sr = librosa.load(wave_file_path, sr=16000)
##wav_2, sr = soundfile.read(wave_file_path)
#
#duration = wav_1.shape[0] / sr
#print("duration: {}".format(duration))
#print(np.all(np.equal(wav_1, wav_2)))
##print("wav: {}".format(wav_1))
#
#stft_1 = stft(y=wav_1)
#
#stft_2 = librosa.stft(y=wav_2, n_fft=1024, hop_length=256, win_length=1024)
#
#print(np.all(np.equal(stft_1, stft_2)))
##print("stft: {}".format(stft_1))
#
#mel = melspectrogram(wav_1)
#
#print(mel.shape)
#
#time_step = duration / (mel.shape[1] - 1)
#
#print("time_step: {}".format(time_step))
#
#time_stamps = np.arange(mel.shape[1]) * time_step
#
#print(time_stamps)

#=======================================================================================================================

#import tensorflow as tf
#
#max_length = 10
#d_model = 32
#
#lookup_table = np.empty(shape=[max_length, d_model], dtype=np.float32)
#
#pos_enc = np.array([[pos / np.power(10000., 2 * (i // 2) / d_model) for i in range(d_model)] for pos in range(max_length)])
#print("pos enc: {}".format(pos_enc))
#lookup_table[:, 0::2] = np.sin(pos_enc[:, 0::2])
#lookup_table[:, 1::2] = np.cos(pos_enc[:, 1::2])
#print("lookup table: {}".format(lookup_table))
#lookup_table = tf.convert_to_tensor(lookup_table)
#
#a = tf.nn.embedding_lookup(lookup_table, [0, 1, 2])
#
#sess = tf.Session()
#
#print(sess.run(a))

#=======================================================================================================================

#import numpy as np
#import tensorflow as tf
#
#from ops import multi_stack_lstm
#
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
#batch_size = 5
#time_steps = 100
#hidden_size = 256
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
#print(length, speaker_embedding, content_embedding, landmark_label)
##content_embedding.set_shape([None, None, 64])
#content_embedding = tf.pad(content_embedding, paddings=[(0, 0), (0, 18), (0, 0)])
#stacked_lstm = multi_stack_lstm(hidden_size=hidden_size, num_lstms=3)
#initial_state = state = stacked_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
#
#print(initial_state)
#
#outputs, states = tf.nn.dynamic_rnn(stacked_lstm, content_embedding, initial_state=initial_state, time_major=False, sequence_length=length)
#outputs = outputs[:, 18:, :]
#
#mask = tf.sequence_mask(lengths=length, maxlen=tf.shape(outputs)[1])
#padding_mask = tf.tile(tf.expand_dims(mask, axis=1), multiples=[1, tf.shape(mask)[1], 1])
#
#print(outputs, states)
#
#initializer = tf.global_variables_initializer()
#
#i = 0
#with tf.Session() as sess:
#    sess.run(initializer)
#    try:
#        while True:
#            i += 1
#            pd, m, out, l, se, ce, ll = sess.run([padding_mask, mask, outputs, length, speaker_embedding, content_embedding, landmark_label])
#            print(i, l, se.shape, ce.shape, ll.shape, m, pd)
#            print(out.shape)
#    except:
#        print("Training process finished")
#        pass
#

#=======================================================================================================================

#import tensorflow as tf
#
#tf.random.set_random_seed(1000)
#
#Q = tf.random.uniform(shape=[5, 5, 2])
#
#K = tf.random.uniform(shape=[5, 5, 2])
#
#V = tf.random.uniform(shape=[5, 5, 2])
#
#attention_map_logits = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
#attention_map = tf.nn.softmax(attention_map_logits, axis=2)
#out = tf.matmul(attention_map, V)
#
#with tf.Session() as sess:
#    q, k, lg, att, o = sess.run([Q, K, attention_map_logits, attention_map, out])
#    print("q: {}, k: {}, attention map logits: {}, attention map: {}, out: {}".format(q, k, lg, att, o))

#=======================================================================================================================

#import numpy as np
#import tensorflow as tf
#
#
#def gather_scatter_example():
#    batch_size = 2
#    seq_len = 5
#    num_classes = 3
#    tf.set_random_seed(23)
#
#    logits = tf.random_normal([batch_size, seq_len, seq_len, num_classes])
#    gather_indices = tf.constant([[0, 1, 1],
#                                  [0, 1, 2],
#                                  [0, 2, 3],
#                                  [1, 0, 1]])
#
#    with tf.Session() as sess:
#        g = tf.gather_nd(logits, gather_indices)
#        results = sess.run([logits, tf.shape(logits), gather_indices, g])
#
#    for r in results:
#        print(r)
#
#gather_scatter_example()

#=======================================================================================================================

#import tensorflow as tf
#
#indices = tf.constant([0, 1, 1])
#x = tf.constant([[1, 2, 3],
#                 [4, 5, 6],
#                 [7, 8, 9]])
#
#result = tf.gather(x, indices, axis=1)
#
#with tf.Session() as sess:
#    selection = sess.run(result)
#    print(selection)

#=======================================================================================================================

import tensorflow as tf
from ops import construct_padding_mask_from_sequence_mask, mlp, encoder, embedding_layer, positional_encoding, \
    encoder_layer, feed_forward, fully_connected, layer_norm, multi_stack_lstm, lstm_cell, multihead_attention, \
    scaled_dot_product_attention, discriminator, construct_lookup_table, speaker_aware_animation, speech_content_animation
from losses import l2_loss, graph_laplacian, discriminator_loss, adversarial_loss
# construct_padding_mask_from_sequence_mask

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

batch_size = 5
hidden_size = 256
d_model = 32

dataset = tf.data.TFRecordDataset(["sequence_example.tfrecord"])
dataset = dataset.map(_parse_single_sequence_example)
dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=([], [204], [256], [None, 64], [None, 204]))
#dataset = dataset.batch(1)
dataset = dataset.shuffle(100)
dataset = dataset.repeat(1)

iterator = dataset.make_one_shot_iterator()
length, static_landmark, speaker_embedding, content_embedding, landmark_label = iterator.get_next()
print(length, static_landmark, speaker_embedding, content_embedding, landmark_label)

sequence_mask = tf.sequence_mask(lengths=length, maxlen=tf.shape(content_embedding)[1])

mask = construct_padding_mask_from_sequence_mask(sequence_mask=sequence_mask)

lookup_table = construct_lookup_table(d_model=32, max_length=5000)
delta_q_t, content_states = speech_content_animation(content_embedding=content_embedding, input_landmarks=static_landmark,
                                                     length=length,
                                                     batch_size=batch_size, hidden_size=hidden_size, num_lstms=3,
                                                     is_training=True)

delta_p_t, speaker_states = speaker_aware_animation(content_embedding=content_embedding, speaker_embedding=speaker_embedding,
                                                    input_landmarks=static_landmark, length=length, lookup_table=lookup_table,
                                                    batch_size=batch_size, d_model=d_model, hidden_size=hidden_size,
                                                    num_lstms=3, tau_comma=256, num_layers=2, num_heads=8, d_ff=2048,
                                                    is_training=True)

output_landmarks = tf.tile(tf.expand_dims(static_landmark, 1), multiples=[1, tf.shape(content_embedding)[1], 1]) + \
    delta_q_t + delta_p_t

real_scores = discriminator(landmarks=landmark_label, hidden_states=speaker_states, speaker_embedding=speaker_embedding,
                            length=length, lookup_table=lookup_table, input_landmarks=static_landmark, batch_size=batch_size,
                            d_model=d_model, num_layers=2, num_heads=8, d_ff=2048, reuse=False)

synthesized_scores = discriminator(landmarks=output_landmarks, hidden_states=speaker_states, speaker_embedding=speaker_embedding,
                                   length=length, lookup_table=lookup_table, input_landmarks=static_landmark, batch_size=batch_size,
                                   d_model=d_model, num_layers=2, num_heads=8, d_ff=2048, reuse=True)

square_loss = l2_loss(predict=output_landmarks, groundtruth=landmark_label, sequence_mask=sequence_mask)
graph_loss = graph_laplacian(predict=output_landmarks, groundtruth=landmark_label, sequence_mask=sequence_mask,
                             is_2d=False)
dis_loss = discriminator_loss(real_scores=real_scores, synthesized_scores=synthesized_scores, sequence_mask=sequence_mask)
adv_loss = adversarial_loss(synthesized_scores=synthesized_scores, sequence_mask=sequence_mask)

initializer = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(initializer)
    sm, m, lt = sess.run([sequence_mask, mask, lookup_table])
    print("sm: {}, m: {}, lt: {}".format(sm, m, lt))

    dqt, cs = sess.run([delta_q_t, content_states])
    print(dqt.shape, len(cs))

    dpt, ss, olm, r_sc, s_sc, sl, gl, dl, al = sess.run([delta_p_t, speaker_states, output_landmarks, real_scores, synthesized_scores, square_loss, graph_loss, dis_loss, adv_loss])
    print(dpt.shape, ss.shape)
    print(olm.shape)
    print(r_sc.shape, s_sc.shape)
    print(sl, gl, dl, al)
