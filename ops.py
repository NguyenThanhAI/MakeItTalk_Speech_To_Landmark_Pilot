import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


def lstm_cell(hidden_size):
    return tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)


@slim.add_arg_scope
def multi_stack_lstm(hidden_size, num_lstms=3, scope=None):
    with tf.variable_scope(scope, "multi_stack_lstm") as sc:
        net = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(hidden_size=hidden_size) for _ in range(num_lstms)])

        #net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        return net


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


@slim.add_arg_scope
def fully_connected(inputs, num_outputs, dropout_rate=None, scope=None,
                    outputs_collections=None, activation_fn=tf.nn.relu):
    with tf.variable_scope(scope, "fully_connected", [inputs]) as sc:
        net = slim.fully_connected(inputs=inputs, num_outputs=num_outputs, activation_fn=None)

        if activation_fn is not None:
            net = activation_fn(net)

        if dropout_rate:
            net = slim.dropout(net, keep_prob=dropout_rate)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


@slim.add_arg_scope
def layer_norm(inputs, scope=None, outputs_collections=None):
    with tf.variable_scope(scope, "layer_normalization", [inputs]) as sc:
        net = slim.layer_norm(inputs=inputs, center=True, scale=True)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


@slim.add_arg_scope
def embedding_layer(inputs, d_model=32, dropout_rate=None, scope="embedding_layer", outputs_collections=None):
    with tf.variable_scope(scope, "embedding_layer", [inputs]) as sc:
        net = fully_connected(inputs=inputs, num_outputs=d_model, dropout_rate=dropout_rate,
                              scope="embedding_fc", activation_fn=None)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


def construct_lookup_table(d_model=32, max_length=10000):
    lookup_table = np.empty(shape=[max_length, d_model], dtype=np.float32)

    pos_enc = np.array([[pos / np.power(10000., 2 * (i // 2) / d_model) for i in range(d_model)] for pos in range(max_length)])

    lookup_table[:, 0::2] = np.sin(pos_enc[:, 0::2])
    lookup_table[:, 1::2] = np.cos(pos_enc[:, 1::2])

    lookup_table = tf.convert_to_tensor(lookup_table, dtype=tf.float32)

    return lookup_table


def positional_encoding(lookup_table: tf.Tensor, pos_indices: tf.Tensor):
    return tf.nn.embedding_lookup(params=lookup_table, ids=pos_indices)


@slim.add_arg_scope
def feed_forward(inputs, d_ff=2048, d_model=32, dropout_rate=None, scope="feed_forward", outputs_collections=None):
    with tf.variable_scope(scope, "feed_forward", [inputs]) as sc:
        net = fully_connected(inputs=inputs, num_outputs=d_ff, dropout_rate=dropout_rate, scope="intermediate_fc")

        net = fully_connected(inputs=net, num_outputs=d_model, dropout_rate=dropout_rate, scope="output_fc", activation_fn=None)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


@slim.add_arg_scope
def mlp(inputs, num_outputs=204, scope="mlp", dropout_rate=None, outputs_collections=None):
    with tf.variable_scope(scope, "mlp", [inputs]) as sc:
        net = fully_connected(inputs=inputs, num_outputs=512, dropout_rate=dropout_rate, scope="fc_1")

        net = fully_connected(inputs=net, num_outputs=256, dropout_rate=dropout_rate, scope="fc_2")

        net = fully_connected(inputs=net, num_outputs=num_outputs, dropout_rate=dropout_rate, scope="fc_3", activation_fn=None)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


def construct_padding_mask_from_sequence_mask(sequence_mask: tf.Tensor):
    seq_len = tf.shape(sequence_mask)[1]
    mask = tf.tile(tf.expand_dims(sequence_mask, axis=1), multiples=[1, seq_len, 1])

    return mask


def scaled_dot_product_attention(Q, K, V, mask=None, scope="scaled_dot_product_attention"):
    # Q: [n_heads * batch_size, max_seq_len, d_model / n_heads]
    with tf.variable_scope(scope, "scaled_dot_product_attention", [Q, K, V]) as sc:
        d = Q.shape[-1]
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1]

        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) # [n_heads * batch_size, max_seq_len, max_seq_len]
        out = tf.divide(out, tf.sqrt(tf.cast(d, tf.float32)))

        mask = tf.cast(mask, dtype=tf.float32)

        if mask is not None:
            out = tf.multiply(out, mask) + (1.0 - mask) * (1e-10)

        out = tf.nn.softmax(out)
        out = tf.matmul(out, V)

    return out


def multihead_attention(queries, memories=None, d_model=32, num_heads=8, mask=None, scope="multihead_attention"):

    with tf.variable_scope(scope, "multihead_attention", [queries, memories, mask]) as sc:
        if memories is None:
            memories = queries

        with tf.variable_scope("linear"):
            Q = fully_connected(inputs=queries, num_outputs=d_model, dropout_rate=None, scope="linear_q")
            K = fully_connected(inputs=memories, num_outputs=d_model, dropout_rate=None, scope="linear_k")
            V = fully_connected(inputs=memories, num_outputs=d_model, dropout_rate=None, scope="linear_v")
        print("Q: {}, K: {}, V: {}".format(Q, K, V))
        Q_split = tf.concat(tf.split(value=Q, num_or_size_splits=num_heads, axis=2), axis=0)
        K_split = tf.concat(tf.split(value=K, num_or_size_splits=num_heads, axis=2), axis=0)
        V_split = tf.concat(tf.split(value=V, num_or_size_splits=num_heads, axis=2), axis=0)
        mask_split = tf.tile(mask, [num_heads, 1, 1])

        out = scaled_dot_product_attention(Q=Q_split, K=K_split, V=V_split, mask=mask_split)

        out = tf.concat(tf.split(value=out, num_or_size_splits=num_heads, axis=0), axis=2)

    return out


@slim.add_arg_scope
def encoder_layer(inputs, input_mask, dropout_rate=None, d_ff=2048, d_model=32, num_heads=8, scope="encoder_layer",
                  outputs_collections=None):
    with tf.variable_scope(scope, "encoder_layer", [inputs]) as sc:
        net = inputs
        net = layer_norm(net + multihead_attention(queries=net, d_model=d_model, num_heads=num_heads, mask=input_mask),
                         scope="layer_normalization_1")
        net = layer_norm(net + feed_forward(inputs=net, d_ff=d_ff, d_model=d_model, dropout_rate=dropout_rate,
                                            scope="feed_forward"),
                         scope="layer_normalization_2")

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


@slim.add_arg_scope
def encoder(inputs, input_mask, num_layers=2, dropout_rate=None, d_ff=2048, d_model=32, num_heads=8,
            scope="encoder", outputs_collections=None):
    with tf.variable_scope(scope, "encoder", [inputs, input_mask]) as sc:
        net = inputs
        for i in range(num_layers):
            net = encoder_layer(inputs=net, input_mask=input_mask, dropout_rate=dropout_rate,
                                d_ff=d_ff, d_model=d_model, num_heads=num_heads, scope="encoder_layer_{}".format(i))

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


def discriminator(landmarks: tf.Tensor, hidden_states: tf.Tensor, speaker_embedding: tf.Tensor, length: tf.Tensor,
                  lookup_table: tf.Tensor, input_landmarks: tf.Tensor,
                  batch_size=16, d_model=32, num_layers=2, dropout_rate=None, d_ff=2048, num_heads=8,
                  scope="discriminator", reuse=None):
    with tf.variable_scope(scope, "discriminator", [landmarks, hidden_states, speaker_embedding], reuse=reuse) as sc:
        end_points_collection = sc.name + "_end_points"

        with slim.arg_scope([slim.dropout], is_training=True), \
             slim.arg_scope([fully_connected, layer_norm, embedding_layer, feed_forward, mlp, encoder_layer,
                             encoder], outputs_collections=end_points_collection), \
             slim.arg_scope([fully_connected, embedding_layer, feed_forward, mlp, encoder_layer, encoder],
                            dropout_rate=dropout_rate):
            speaker_embedding = tf.tile(tf.expand_dims(speaker_embedding, axis=1),
                                        multiples=[1, tf.shape(landmarks)[1], 1])

            input_attention = tf.concat([landmarks, hidden_states, speaker_embedding], axis=2)

            position_index = tf.tile(tf.expand_dims(tf.range(tf.shape(input_attention)[1]), axis=0), multiples=[batch_size, 1])

            position_index = tf.reshape(position_index, shape=[-1])

            sequence_mask = tf.sequence_mask(lengths=length, maxlen=tf.reduce_max(length))

            sequence_mask = construct_padding_mask_from_sequence_mask(sequence_mask)

            pos_embedding = positional_encoding(lookup_table=lookup_table, pos_indices=position_index)

            #input_attention = tf.reshape(input_attention, shape=[-1, tf.shape(input_attention)[-1]])
            input_embedding = embedding_layer(inputs=input_attention, d_model=d_model, scope="embedding_layer")
            pos_embedding = tf.reshape(pos_embedding, shape=[tf.shape(input_attention)[0], tf.shape(input_attention)[1], input_embedding.get_shape()[-1]])
            input_encoder = input_embedding + pos_embedding

            output_attention = encoder(inputs=input_encoder, input_mask=sequence_mask, num_layers=num_layers,
                                       d_ff=d_ff, d_model=d_model, num_heads=num_heads, scope="encoder")

            #output_attention = tf.reshape(output_attention, shape=[-1, d_model])

            input_landmarks = tf.tile(tf.expand_dims(input_landmarks, axis=1), multiples=[1, tf.reduce_max(length), 1])

            #input_landmarks = tf.reshape(input_landmarks, [-1, tf.shape(input_landmarks)[-1]])

            input_mlp = tf.concat(values=[input_landmarks, output_attention], axis=-1)

            scores = mlp(inputs=input_mlp, num_outputs=1)

            #scores = tf.reshape(scores, shape=[-1, tf.reduce_max(length)])

            scores = tf.nn.sigmoid(scores)

            return scores


def speech_content_animation(content_embedding: tf.Tensor, input_landmarks: tf.Tensor, length: tf.Tensor,
                             batch_size=16,
                             hidden_size=256, num_lstms=3, tau=18, dropout_rate=None,
                             scope="speech_content_animation", is_training=True, is_2d=True):
    with tf.variable_scope(scope, "speech_content_animation", [content_embedding]) as sc:
        end_points_collection = sc.name + "_end_points"

        with slim.arg_scope([slim.dropout], is_training=is_training), \
            slim.arg_scope([fully_connected, mlp], outputs_collections=end_points_collection), \
            slim.arg_scope([mlp, fully_connected], dropout_rate=dropout_rate):

            content_embedding = tf.pad(content_embedding, paddings=[(0, 0), (0, tau), (0, 0)])
            plus_length = tf.add(length, tau * tf.ones_like(length))
            print("content_embedding: {}".format(content_embedding))
            stacked_lstm = multi_stack_lstm(hidden_size=hidden_size, num_lstms=num_lstms, scope="multi_stack_lstm")
            initial_state = stacked_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

            outputs, states = tf.nn.dynamic_rnn(stacked_lstm, content_embedding,
                                                initial_state=initial_state, time_major=False, sequence_length=plus_length)
            print("outputs: {}".format(outputs))
            outputs = outputs[:, tau:, :]

            #outputs = tf.reshape(outputs, shape=[-1, hidden_size])

            input_landmarks = tf.tile(tf.expand_dims(input_landmarks, axis=1), multiples=[1, tf.reduce_max(length), 1])

            #input_landmarks = tf.reshape(input_landmarks, [-1, input_landmarks.get_shape()[-1]])

            input_mlp = tf.concat(values=[input_landmarks, outputs], axis=-1)
            print("input_mlp: {}".format(input_mlp))
            if is_2d:
                num_outputs = 136
            else:
                num_outputs = 204
            outputs = mlp(inputs=input_mlp, num_outputs=num_outputs)

            #outputs = tf.reshape(outputs, shape=[-1, tf.reduce_max(length), tf.shape(input_landmarks)[-1]])

        return outputs, states


def speaker_aware_animation(content_embedding: tf.Tensor, speaker_embedding: tf.Tensor, input_landmarks: tf.Tensor,
                            length: tf.Tensor,
                            lookup_table: tf.Tensor, batch_size=16, d_model=32,
                            hidden_size=256, num_lstms=3, tau_comma=256, dropout_rate=None,
                            num_layers=2, num_heads=8, d_ff=2048,
                            scope="speaker_aware_animation", is_training=True, is_2d=True):
    with tf.variable_scope(scope, "speaker_aware_animation", [content_embedding, speaker_embedding]) as sc:
        end_points_collection = sc.name + "_end_points"

        with slim.arg_scope([slim.dropout], is_training=is_training), \
            slim.arg_scope([fully_connected, layer_norm, embedding_layer, feed_forward, mlp, encoder_layer,
                            encoder], outputs_collections=end_points_collection), \
            slim.arg_scope([fully_connected, embedding_layer, feed_forward, mlp, encoder_layer, encoder],
                           dropout_rate=dropout_rate):

            content_embedding = tf.pad(content_embedding, [(0, 0), (0, tau_comma), (0, 0)]) # Pad thêm 256 time steps để đưa vào mạng lstm
            plus_length = tf.add(length, tau_comma * tf.ones_like(length))

            stacked_lstm = multi_stack_lstm(hidden_size=hidden_size, num_lstms=num_lstms, scope="multi_stack_lstm")
            initial_state = stacked_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

            out_lstm, states = tf.nn.dynamic_rnn(stacked_lstm, content_embedding,
                                                 initial_state=initial_state, time_major=False, sequence_length=plus_length)
            out_lstm = out_lstm[:, tau_comma:, :] # Lấy times step vị trí 256 trở đi

            tiled_speaker_embedding = tf.tile(tf.expand_dims(speaker_embedding, axis=1), multiples=[1, tf.shape(out_lstm)[1], 1])

            input_attention = tf.concat([out_lstm, tiled_speaker_embedding], axis=2)

            position_index = tf.tile(tf.expand_dims(tf.range(tf.shape(input_attention)[1]), axis=0), multiples=[batch_size, 1])

            position_index = tf.reshape(position_index, shape=[-1])

            sequence_mask = tf.sequence_mask(lengths=length, maxlen=tf.reduce_max(length))

            sequence_mask = construct_padding_mask_from_sequence_mask(sequence_mask)

            pos_embedding = positional_encoding(lookup_table=lookup_table, pos_indices=position_index)

            #input_attention = tf.reshape(input_attention, shape=[-1, input_attention.get_shape()[-1]])
            input_embedding = embedding_layer(inputs=input_attention, d_model=d_model, scope="embedding_layer")
            pos_embedding = tf.reshape(pos_embedding, shape=[tf.shape(input_attention)[0], tf.shape(input_attention)[1], input_embedding.get_shape()[-1]])
            input_encoder = input_embedding + pos_embedding

            output_attention = encoder(inputs=input_encoder, input_mask=sequence_mask, d_model=d_model,
                                       num_layers=num_layers, d_ff=d_ff, num_heads=num_heads)

            #output_attention = tf.reshape(output_attention, shape=[-1, d_model])

            input_landmarks = tf.tile(tf.expand_dims(input_landmarks, axis=1), multiples=[1, tf.reduce_max(length), 1])

            #input_landmarks = tf.reshape(input_landmarks, [-1, input_landmarks.get_shape()[-1]])

            input_mlp = tf.concat(values=[input_landmarks, output_attention], axis=-1)
            if is_2d:
                num_outputs = 136
            else:
                num_outputs = 204
            outputs = mlp(inputs=input_mlp, num_outputs=num_outputs)

            #outputs = tf.reshape(outputs, shape=[-1, tf.reduce_max(length), input_landmarks.get_shape()[-1]])

        return outputs, out_lstm



#batch_size = 5
#time_steps = 100
#hidden_size = 256
#stacked_lstm = multi_stack_lstm(hidden_size=hidden_size, num_lstms=3)
#initial_state = state = stacked_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
#
#print(initial_state)
#inputs = tf.random.uniform(shape=[batch_size, time_steps, hidden_size], maxval=1., dtype=tf.float32)
#
#length_of_input = length(inputs)
#
#outputs, states = tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=initial_state, time_major=False, sequence_length=length(inputs))
#
#print(outputs, states)
#
#initializer = tf.global_variables_initializer()
#
#with tf.Session() as sess:
#    sess.run(initializer)
#    l, out, st = sess.run([length_of_input, outputs, states])
#    print("length of input: {}, out: {}, st: {}".format(l, out, st))
