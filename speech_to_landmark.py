import os
from datetime import datetime
import time

import numpy as np

import tensorflow as tf
from read_dataset_utils import get_batch
from ops import construct_padding_mask_from_sequence_mask, construct_lookup_table, \
    speech_content_animation, speaker_aware_animation, discriminator
from losses import l2_loss, graph_laplacian, discriminator_loss, adversarial_loss


class SpeechToLandmarkConfig(object):
    def __init__(self, batch_size=32, hidden_size=256, d_model=32, max_length=1000, num_lstms=3,
                 tau=18, tau_comma=256, num_layers=2, num_heads=8, d_ff=2048, dataset_dir=None,
                 speech_content_checkpoint=None, speaker_aware_checkpoint=None, discriminator_checkpoint=None,
                 speech_content_model_dir=None, speaker_aware_model_dir=None, discriminator_model_dir=None,
                 num_epochs=100, dropout_rate=None, is_2d=True, summary_dir="summary",
                 summary_frequency=10, save_network_frequency=10000, is_training=True, optimizer="adam",
                 per_process_gpu_memory_fraction=1.0, learning_rate=5e-4, learning_rate_decay_type="constant",
                 decay_steps=10000, decay_rate=0.9, momentum=0.9, lambda_c=1., lambda_s=1., miu_s=0.001,
                 use_speaker_aware=False, is_loadmodel=True):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.max_length = max_length
        self.num_lstms = num_lstms
        self.tau = tau
        self.tau_comma = tau_comma
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dataset_dir = dataset_dir
        self.speech_content_checkpoint = speech_content_checkpoint
        self.speaker_aware_checkpoint = speaker_aware_checkpoint
        self.discriminator_checkpoint = discriminator_checkpoint
        self.speech_content_model_dir = speech_content_model_dir
        self.speaker_aware_model_dir = speaker_aware_model_dir
        self.discriminator_model_dir = discriminator_model_dir
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.is_2d = is_2d
        self.summary_dir = summary_dir
        self.summary_frequency = summary_frequency
        self.save_network_frequency = save_network_frequency
        self.is_training = is_training
        self.optimizer = optimizer
        self.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        self.learning_rate = learning_rate
        self.learning_rate_decay_type = learning_rate_decay_type
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.miu_s = miu_s
        self.use_speaker_aware = use_speaker_aware
        self.is_loadmodel = is_loadmodel


class SpeechToLandmark(object):
    def __init__(self, config: SpeechToLandmarkConfig):
        self.config = config
        tf.reset_default_graph()

        self.length, self.static_landmark, self.speaker_embedding, self.content_embedding, self.landmark_label, self.epoch_now = get_batch(tfrecord_path=self.config.dataset_dir,
                                                                                                                                           batch_size=self.config.batch_size,
                                                                                                                                           num_epochs=self.config.num_epochs)

        self.sequence_mask = tf.sequence_mask(lengths=self.length, maxlen=tf.shape(self.content_embedding)[1])
        self.mask = construct_padding_mask_from_sequence_mask(sequence_mask=self.sequence_mask)

        self.lookup_table = construct_lookup_table(d_model=self.config.d_model, max_length=self.config.max_length)

        self.delta_q_t, self.content_states = speech_content_animation(content_embedding=self.content_embedding,
                                                                       input_landmarks=self.static_landmark,
                                                                       length=self.length,
                                                                       batch_size=self.config.batch_size,
                                                                       hidden_size=self.config.hidden_size,
                                                                       num_lstms=self.config.num_lstms,
                                                                       tau=self.config.tau,
                                                                       dropout_rate=self.config.dropout_rate,
                                                                       is_training=self.config.is_training,
                                                                       scope="speech_content_animation",
                                                                       is_2d=self.config.is_2d)

        if self.config.use_speaker_aware:

            self.delta_p_t, self.speaker_states = speaker_aware_animation(content_embedding=self.content_embedding,
                                                                          speaker_embedding=self.speaker_embedding,
                                                                          input_landmarks=self.static_landmark,
                                                                          length=self.length, lookup_table=self.lookup_table,
                                                                          batch_size=self.config.batch_size,
                                                                          d_model=self.config.d_model,
                                                                          hidden_size=self.config.hidden_size,
                                                                          num_lstms=self.config.num_lstms,
                                                                          tau_comma=self.config.tau_comma,
                                                                          dropout_rate=self.config.dropout_rate,
                                                                          num_layers=self.config.num_layers,
                                                                          num_heads=self.config.num_heads,
                                                                          d_ff=self.config.d_ff,
                                                                          scope="speaker_aware_animation",
                                                                          is_training=self.config.is_training,
                                                                          is_2d=self.config.is_2d)
        if self.config.use_speaker_aware:
            self.output_landmarks = tf.tile(tf.expand_dims(self.static_landmark, axis=1), multiples=[1, tf.shape(self.content_embedding)[1], 1]) + \
                self.delta_q_t + self.delta_p_t
        else:
            self.output_landmarks = tf.tile(tf.expand_dims(self.static_landmark, axis=1), multiples=[1, tf.shape(self.content_embedding)[1], 1]) + \
                self.delta_q_t

        if self.config.use_speaker_aware:
            self.real_scores = discriminator(landmarks=self.landmark_label, hidden_states=self.speaker_states,
                                             speaker_embedding=self.speaker_embedding, length=self.length,
                                             lookup_table=self.lookup_table, input_landmarks=self.static_landmark,
                                             batch_size=self.config.batch_size,
                                             d_model=self.config.d_model, num_layers=self.config.num_layers,
                                             dropout_rate=self.config.dropout_rate, d_ff=self.config.d_ff,
                                             num_heads=self.config.num_heads, scope="discriminator", reuse=False)
            self.synthesized_scores = discriminator(landmarks=self.output_landmarks, length=self.length,
                                                    lookup_table=self.lookup_table, hidden_states=self.speaker_states,
                                                    speaker_embedding=self.speaker_embedding, input_landmarks=self.static_landmark,
                                                    batch_size=self.config.batch_size, d_model=self.config.d_model,
                                                    num_layers=self.config.num_layers, dropout_rate=self.config.dropout_rate,
                                                    d_ff=self.config.d_ff, num_heads=self.config.num_heads, scope="discriminator", reuse=True)

        self.global_step = tf.train.get_or_create_global_step()
        self.global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + \
                                tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        self.speech_content_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                          scope="speech_content_animation")
        self.speech_content_saver = tf.train.Saver(var_list=self.speech_content_variables + [self.global_step])

        if self.config.use_speaker_aware:
            self.speaker_aware_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                             scope="speaker_aware_animation")
            self.speaker_aware_saver = tf.train.Saver(var_list=self.speaker_aware_variables + [self.global_step])
            self.discriminator_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                             scope="discriminator")
            self.discriminator_saver = tf.train.Saver(var_list=self.discriminator_variables + [self.global_step])

        with tf.name_scope("loss"):
            self.square_loss = l2_loss(predict=self.output_landmarks, groundtruth=self.landmark_label,
                                       sequence_mask=self.sequence_mask)
            self.graph_loss = graph_laplacian(predict=self.output_landmarks, groundtruth=self.landmark_label,
                                              sequence_mask=self.sequence_mask)
            tf.summary.scalar("square loss", self.square_loss)
            tf.summary.scalar("graph loss", self.graph_loss)

            if self.config.use_speaker_aware:
                self.dis_loss = discriminator_loss(real_scores=self.real_scores,
                                                   synthesized_scores=self.synthesized_scores,
                                                   sequence_mask=self.sequence_mask)
                self.adv_loss = adversarial_loss(synthesized_scores=self.synthesized_scores,
                                                 sequence_mask=self.sequence_mask)
                tf.summary.scalar("dis loss", self.dis_loss)
                tf.summary.scalar("adv loss", self.adv_loss)


            if self.config.use_speaker_aware:
                self.speaker_aware_loss = self.square_loss + self.config.lambda_s * self.graph_loss + self.config.miu_s * self.adv_loss
                tf.summary.scalar("speaker aware loss", self.speaker_aware_loss)
            else:
                self.speech_content_loss = self.square_loss + self.config.lambda_c * self.graph_loss
                tf.summary.scalar("speech content loss", self.speech_content_loss)

        with tf.name_scope("optimizer"):
            if self.config.learning_rate_decay_type == "constant":
                self.learning_rate = self.config.learning_rate
            elif self.config.learning_rate_decay_type == "piecewise_constant":
                self.learning_rate = tf.train.piecewise_constant(x=self.global_step,
                                                                 boundaries=[20000, 200000, 500000],
                                                                 values=[5e-4, 2.5e-4, 1e-4, 5e-5])
            elif self.config.learning_rate_decay_type == "exponential_decay":
                self.learning_rate = tf.train.exponential_decay(learning_rate=self.config.learning_rate,
                                                                global_step=self.global_step,
                                                                decay_steps=self.config.decay_steps,
                                                                decay_rate=self.config.decay_rate,
                                                                staircase=True)
            elif self.config.learning_rate_decay_type == "linear_cosine_decay":
                self.learning_rate = tf.train.linear_cosine_decay(learning_rate=self.config.learning_rate,
                                                                  global_step=self.global_step,
                                                                  decay_steps=self.config.decay_steps)
            tf.summary.scalar("learning rate", self.learning_rate)
            if self.config.optimizer.lower() == "adam":
                if not self.config.use_speaker_aware:
                    self.speech_content_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, beta1=0.5, beta2=0.999)
                else:
                    self.speaker_aware_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, beta1=0.5, beta2=0.999)
                    self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, beta1=0.5, beta2=0.999)
            elif self.config.optimizer.lower() == "rms" or self.config.optimizer.lower() == "rmsprop":
                if not self.config.use_speaker_aware:
                    self.speech_content_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                else:
                    self.speaker_aware_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                    self.discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            elif self.config.optimizer.lower() == "momentum":
                if not self.config.use_speaker_aware:
                    self.speech_content_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                                momentum=self.config.momentum)
                else:
                    self.speaker_aware_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                              momentum=self.config.momentum)
                    self.discriminator_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                              momentum=self.config.momentum)
            elif self.config.optimizer.lower() == "grad_descent":
                if not self.config.use_speaker_aware:
                    self.speech_content_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
                else:
                    self.speaker_aware_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
                    self.discriminator_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            else:
                raise ValueError("Optimizer {} was not recognized".format(self.config.optimizer))

            if not self.config.use_speaker_aware:
                self.speech_content_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="speech_content_animation")
                with tf.control_dependencies(self.speech_content_update_ops):
                    self.speech_content_train_op = self.speech_content_optimizer.minimize(self.speech_content_loss,
                                                                                          global_step=self.global_step,
                                                                                          var_list=self.speech_content_variables)
            else:
                self.speaker_aware_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="speaker_aware_animation")
                with tf.control_dependencies(self.speaker_aware_update_ops):
                    self.speaker_aware_train_op = self.speaker_aware_optimizer.minimize(self.speaker_aware_loss,
                                                                                        global_step=self.global_step,
                                                                                        var_list=self.speaker_aware_variables)
                self.discriminator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="discriminator")
                with tf.control_dependencies(self.discriminator_update_ops):
                    self.discriminator_train_op = self.discriminator_optimizer.minimize(self.dis_loss,
                                                                                        global_step=self.global_step,
                                                                                        var_list=self.discriminator_variables)

        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.config.per_process_gpu_memory_fraction)
        self.config_gpu = tf.ConfigProto(gpu_options=self.gpu_options)
        self.sess = tf.Session(config=self.config_gpu)

        self.merged = tf.summary.merge_all()

        if self.config.is_loadmodel:
            self.writer = tf.summary.FileWriter(logdir=self.config.summary_dir)
        else:
            self.writer = tf.summary.FileWriter(logdir=self.config.summary_dir, graph=self.sess.graph)

        self.restore_or_initialize_network(speech_content_checkpoint=self.config.speech_content_checkpoint,
                                           speaker_aware_checkpoint=self.config.speaker_aware_checkpoint,
                                           discriminator_checkpoint=self.config.discriminator_checkpoint)

    def restore_or_initialize_network(self, speech_content_checkpoint=None, speaker_aware_checkpoint=None, discriminator_checkpoint=None):
        self.sess.run(tf.global_variables_initializer())
        if not self.config.use_speaker_aware:
            if self.config.is_loadmodel:
                if speech_content_checkpoint is not None:
                    self.speech_content_saver.restore(sess=self.sess, save_path=os.path.join(self.config.speech_content_model_dir, speech_content_checkpoint))
                else:
                    self.speech_content_saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=self.config.speech_content_model_dir))
                print("Successfully load speech content animation model at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            else:
                print("Successfully initialize speech content animation model model at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        else:
            assert self.config.speech_content_model_dir is not None
            if speech_content_checkpoint is not None:
                self.speech_content_saver.restore(sess=self.sess,
                                                  save_path=os.path.join(self.config.speech_content_model_dir, speech_content_checkpoint))
            else:
                self.speech_content_saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=self.config.speech_content_model_dir))
            print("Successfully load speech content animation model at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            if self.config.is_loadmodel:
                if speaker_aware_checkpoint is not None:
                    self.speaker_aware_saver.restore(sess=self.sess, save_path=os.path.join(self.config.speaker_aware_checkpoint, speaker_aware_checkpoint))
                else:
                    self.speaker_aware_saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=self.config.speaker_aware_model_dir))
                if discriminator_checkpoint is not None:
                    self.discriminator_saver.restore(sess=self.sess, save_path=os.path.join(self.config.discriminator_model_dir, discriminator_checkpoint))
                else:
                    self.discriminator_saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=self.config.discriminator_model_dir))
                print("Successfully load speaker aware animation and discriminator model at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            else:
                print("Successfully initialize speaker aware animation and discriminator model at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    def get_global_step(self):
        return tf.train.global_step(sess=self.sess, global_step_tensor=self.global_step)

    def save_network(self, global_step):
        if not self.config.use_speaker_aware:
            if not os.path.exists(self.config.speech_content_model_dir):
                os.makedirs(self.config.speech_content_model_dir, exist_ok=True)
            speech_content_checkpoint_name = "Speech_Content_Animation-" + str(global_step).zfill(7)
            self.speech_content_saver.save(sess=self.sess, save_path=os.path.join(self.config.speech_content_model_dir, speech_content_checkpoint_name))
            print("Save speech content animation module at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        else:
            if not os.path.exists(self.config.speaker_aware_model_dir):
                os.makedirs(self.config.speaker_aware_model_dir, exist_ok=True)
            speaker_aware_checkpoint_name = "Speaker_Aware_Animation-" + str(global_step).zfill(7)
            self.speaker_aware_saver.save(sess=self.sess, save_path=os.path.join(self.config.speaker_aware_model_dir, speaker_aware_checkpoint_name))
            if not os.path.exists(self.config.discriminator_model_dir):
                os.makedirs(self.config.discriminator_model_dir, exist_ok=True)
            discriminator_checkpoint_name = "Discriminator-" + str(global_step).zfill(7)
            self.discriminator_saver.save(sess=self.sess, save_path=os.path.join(self.config.discriminator_model_dir, discriminator_checkpoint_name))
            print("Save speaker aware animation module and discriminator at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    def add_summary(self, global_step):

        summary = self.sess.run(self.merged)

        self.writer.add_summary(summary=summary, global_step=global_step)
        self.writer.flush()
        print("Add summary at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    def train(self):
        try:
            while True:
                start = time.time()
                if not self.config.use_speaker_aware:
                    _, cost, epoch = self.sess.run([self.speech_content_train_op, self.speech_content_loss, self.epoch_now])
                else:
                    _, _, speaker_aware_cost, discriminator_cost, epoch = self.sess.run([self.speaker_aware_train_op,
                                                                                         self.discriminator_train_op,
                                                                                         self.speaker_aware_loss,
                                                                                         self.dis_loss])
                global_step = self.get_global_step()

                if global_step % self.config.summary_frequency == 0:
                    self.add_summary(global_step=global_step)

                if global_step % self.config.save_network_frequency == 0:
                    self.save_network(global_step=global_step)

                end = time.time()

                if not self.config.use_speaker_aware:
                    print("Step: {}, speech content loss: {}, epoch: {}, takes time: {}".format(global_step, np.round(cost), epoch, np.round(end - start)))
                else:
                    print("Step: {}, speaker aware loss: {}, discriminator loss: {}, epoch: {}, takes time: {}".format(global_step,
                                                                                                                       np.round(speaker_aware_cost),
                                                                                                                       np.round(discriminator_cost),
                                                                                                                       epoch,
                                                                                                                       np.round(end - start)))
        except tf.errors.OutOfRangeError:
            global_step = self.get_global_step()
            self.save_network(global_step=global_step)
            print("Training process finished")
            pass
