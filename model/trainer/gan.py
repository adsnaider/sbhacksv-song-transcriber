from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from scipy.io.wavfile import write as write_wav

from trainer import audio_models

FLAGS = flags.FLAGS

flags.DEFINE_string('save_model_location', None,
                    "Where we should save the models.")
flags.DEFINE_string('job-dir', None, "Job directory")
flags.DEFINE_string('job_dir', None, "Job directory")

flags.DEFINE_string('genre1', None,
                    'tfrecords file with all the songs from genre 1')
flags.DEFINE_string('genre2', None,
                    'tfrecords file with all the songs from genre 2')

flags.DEFINE_integer('batch_size', 1, 'Batch size...')
flags.DEFINE_integer('train_steps', 100000, 'train steps.')

flags.DEFINE_float('cycle_weight', 10.0, 'Lambda value for the cycle loss.')
flags.DEFINE_float('dlr', 0.001, "Discriminator Learning rate")
flags.DEFINE_float('glr', 0.001, "Generator Learning rate")

# class GAN():

# def __init__(self, gen_net, dis_net):
# self.gen = gen_net
# self.disc = dis_net

# def generate(self, inputs):
# return self.gen(inputs)

# def predict(self, inputs):
# return tf.sigmoid(self.disc(inputs))

# def loss(self, genre_x, genre_y):
# generated_genre_y = self.generate(genre_x)

# real_predictions = self.predict(genre_y)
# fake_predictions = self.predict(generated_genre_y)

# expected_real = tf.zeros_like(real_predictions)
# expected_fake = tf.ones_like(fake_predictions)

# d_loss = tf.cast(
# tf.losses.mean_squared_error(
# tf.concat([expected_real, expected_fake], axis=0),
# tf.concat([real_predictions, fake_predictions], axis=0)),
# dtype=tf.float64)

# g_loss = tf.cast(
# tf.losses.mean_squared_error(1.0 - expected_fake, fake_predictions),
# dtype=tf.float64)

# return d_loss, g_loss, generated_genre_y


def normalize_song(song):
  return (song + 0.5) / 32767.5


def main(argv):
  del argv

  def _parse_func(example_proto):
    features = {
        'song': tf.FixedLenFeature((), tf.string, default_value=""),
        'sample_rate': tf.FixedLenFeature((), tf.int64, default_value=0)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    song = parsed_features['song']
    sample_rate = parsed_features['sample_rate']

    sample_rate = tf.cast(sample_rate, tf.float32)

    song = tf.decode_raw(song, tf.float64)
    song = tf.reshape(song, shape=[-1, 1])
    song = tf.cast(song, dtype=tf.float32)

    div = 10 * 10 * 10
    # Size must be multiple of div
    mod = tf.mod(tf.shape(song)[0], div)
    extra = tf.mod(div - mod, div)
    pad = tf.zeros(dtype=tf.float32, shape=[extra, 1])
    song = tf.concat([song, pad], axis=0)

    return sample_rate, song

  genre1 = tf.data.TFRecordDataset(FLAGS.genre1)
  genre1 = genre1.map(_parse_func)
  # genre1 = genre1.shuffle(buffer_size=100)
  genre1 = genre1.padded_batch(FLAGS.batch_size, padded_shapes=([], [None, 1]))
  genre1 = genre1.repeat()
  genre1_iterator = genre1.make_one_shot_iterator()
  genre1_sample_rates, genre1_songs = genre1_iterator.get_next()

  genre2 = tf.data.TFRecordDataset(FLAGS.genre2)
  genre2 = genre2.map(_parse_func)
  # genre2 = genre2.shuffle(buffer_size=100)
  genre2 = genre2.padded_batch(FLAGS.batch_size, padded_shapes=([], [None, 1]))
  genre2 = genre2.repeat()
  genre2_iterator = genre2.make_one_shot_iterator()
  genre2_sample_rates, genre2_songs = genre2_iterator.get_next()

  g1 = audio_models.AudioEncoderDecoder()
  d1 = audio_models.AudioClassifier()
  g2 = audio_models.AudioEncoderDecoder()
  d2 = audio_models.AudioClassifier()

  g_opt = tf.train.AdamOptimizer(FLAGS.glr)
  d_opt = tf.train.AdamOptimizer(FLAGS.dlr)

  global_step = tf.train.get_or_create_global_step()
  global_step_update = global_step.assign_add(1)

  # d_loss1, g_loss1, generated_genre2 = g1.loss(genre1_songs, genre2_songs)
  # d_loss2, g_loss2, generated_genre1 = g2.loss(genre2_songs, genre1_songs)

  gx2y = g1(genre1_songs)
  gy2x = g2(genre2_songs)

  d1_real = tf.sigmoid(d1(genre2_songs))
  d1_fake = tf.sigmoid(d1(gx2y))

  d2_real = tf.sigmoid(d2(genre1_songs))
  d2_fake = tf.sigmoid(d2(gy2x))

  g1_loss = tf.reduce_mean(tf.square(d1_fake))
  g2_loss = tf.reduce_mean(tf.square(d2_fake))

  d1_loss = tf.reduce_mean(tf.square(d1_real - 1)) + tf.reduce_mean(
      tf.square(d1_fake))
  d2_loss = tf.reduce_mean(tf.square(d2_real - 1)) + tf.reduce_mean(
      tf.square(d2_fake))

  cycle1 = g2(gx2y)
  cycle2 = g1(gy2x)

  cycle_loss1 = tf.reduce_mean(tf.abs(cycle1 - genre1_songs))
  cycle_loss2 = tf.reduce_mean(tf.abs(cycle2 - genre2_songs))

  g1_loss = g1_loss + FLAGS.cycle_weight * (cycle_loss1 + cycle_loss2)
  g2_loss = g2_loss + FLAGS.cycle_weight * (cycle_loss1 + cycle_loss2)

  update_ops = g1.updates + g2.updates + d1.updates + d2.updates
  with tf.control_dependencies(update_ops):
    train_op = tf.group([
        g_opt.minimize(g1_loss, var_list=g1.trainable_variables),
        g_opt.minimize(g2_loss, var_list=g2.trainable_variables),
        d_opt.minimize(d1_loss, var_list=d1.trainable_variables),
        d_opt.minimize(d2_loss, var_list=d2.trainable_variables)
    ])

  tf.summary.scalar('d1_loss', d1_loss)
  tf.summary.scalar('g1_loss', g1_loss)
  tf.summary.scalar('d2_loss', d2_loss)
  tf.summary.scalar('g2_loss', g2_loss)

  tf.summary.audio(
      'genre1_song',
      normalize_song(tf.expand_dims(genre1_songs[0], 0)),
      genre1_sample_rates[0],
      max_outputs=1)
  tf.summary.audio(
      'genre2_song',
      normalize_song(tf.expand_dims(genre2_songs[0], 0)),
      genre2_sample_rates[0],
      max_outputs=1)

  tf.summary.audio(
      'gen1to2',
      normalize_song(tf.expand_dims(gx2y[0], 0)),
      genre1_sample_rates[0],
      max_outputs=1)
  tf.summary.audio(
      'gen2to1',
      normalize_song(tf.expand_dims(gy2x[0], 0)),
      genre2_sample_rates[0],
      max_outputs=1)

  tf.summary.audio(
      'cycle1',
      normalize_song(tf.expand_dims(cycle1[0], 0)),
      genre1_sample_rates[0],
      max_outputs=1)
  tf.summary.audio(
      'cycle2',
      normalize_song(tf.expand_dims(cycle2[0], 0)),
      genre2_sample_rates[0],
      max_outputs=1)

  if FLAGS.save_model_location:
    saver = tf.train.Saver(var_list=g1.gen.trainable_variables +
                           g2.gen.trainable_variables)

    input1 = tf.placeholder(dtype=tf.float64, shape=[None, None, 1])
    output1 = g1.generate(input1)

    input2 = tf.placeholder(dtype=tf.float64, shape=[None, None, 1])
    output2 = g2.generate(input2)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      checkpoint_dir = os.path.join(FLAGS.job_dir, 'checkpoints')
      saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
      tf.saved_model.simple_save(
          sess,
          os.path.join(FLAGS.save_model_location, '0'),
          inputs={'model_input': input1},
          outputs={'model_output': output1})

      tf.saved_model.simple_save(
          sess,
          os.path.join(FLAGS.save_model_location, '1'),
          inputs={'model_input': input2},
          outputs={'model_output': output2})

  else:
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=os.path.join(FLAGS.job_dir, 'checkpoints')) as sess:
      for i in range(FLAGS.train_steps):
        _, DLoss1, GLoss1, DLoss2, GLoss2 = sess.run(
            [train_op, d1_loss, g1_loss, d2_loss, g2_loss])
        sess.run(global_step_update)
        if i % 30 == 0:
          print("Step {} : D loss : {} : G loss {}".format(
              sess.run(global_step), DLoss1 + DLoss2, GLoss1 + GLoss2))


if __name__ == '__main__':
  flags.mark_flags_as_required(['genre1', 'genre2'])
  app.run(main)
