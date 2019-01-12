from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
from absl import app
from absl import flags
from absl import logging

from trainer import audio_models

# tf.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_string('job-dir', None, "Job directory")

flags.DEFINE_string('genre1', None,
                    'tfrecords file with all the songs from genre 1')
flags.DEFINE_string('genre2', None,
                    'tfrecords file with all the songs from genre 2')

flags.DEFINE_integer('batch_size', 1, 'Batch size...')
flags.DEFINE_integer('train_steps', 100000, 'train steps.')

flags.DEFINE_float('cycle_weight', 10.0, 'Lambda value for the cycle loss.')
flags.DEFINE_float('dlr', 0.001, "Discriminator Learning rate")
flags.DEFINE_float('glr', 0.001, "Generator Learning rate")


class GAN():

  def __init__(self, gen_net, dis_net):
    self.gen = gen_net
    self.disc = dis_net

  def generate(self, inputs):
    return self.gen(inputs)

  def predict(self, inputs):
    return self.disc(inputs)

  def loss(self, genre_x, genre_y):
    generated_genre_y = self.generate(genre_x)

    real_predictions = self.predict(genre_y)
    fake_predictions = self.predict(generated_genre_y)

    expected_real = tf.zeros_like(real_predictions)
    expected_fake = tf.ones_like(fake_predictions)

    d_loss = tf.losses.sigmoid_cross_entropy(
        tf.concat([expected_real, expected_fake], axis=1),
        tf.concat([real_predictions, fake_predictions], axis=1))

    g_loss = tf.losses.sigmoid_cross_entropy(1.0 - expected_fake,
                                             fake_predictions)

    return d_loss, g_loss, generated_genre_y


def main(argv):
  del argv

  def _parse_func(example_proto):
    features = {
        'song': tf.FixedLenFeature((), tf.string, default_value=""),
        'sample_rate': tf.FixedLenFeature((), tf.int64, default_value=0)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["song"], parsed_features["sample_rate"]

  genre1 = tf.data.TFRecordDataset(FLAGS.genre1)
  genre1 = genre1.map(_parse_func)
  genre1 = genre1.shuffle(buffer_size=100)
  genre1 = genre1.batch(FLAGS.batch_size)
  genre1 = genre1.repeat()
  # genre1 = tfe.Iterator(genre1)
  genre1 = genre1.make_one_shot_iterator()
  genre1_songs = genre1.get_next()

  genre2 = tf.data.TFRecordDataset(FLAGS.genre2)
  genre2 = genre2.map(_parse_func)
  genre2 = genre2.shuffle(buffer_size=100)
  genre2 = genre2.batch(FLAGS.batch_size)
  genre2 = genre2.repeat()
  # genre2 = tfe.Iterator(genre2)
  genre2 = genre2.make_one_shot_iterator()
  genre2_songs = genre2.get_next()

  g1 = GAN(audio_models.AudioEncoderDecoder(), audio_models.AudioClassifier())
  g2 = GAN(audio_models.AudioEncoderDecoder(), audio_models.AudioClassifier())

  g_opt = tf.train.AdamOptimizer(FLAGS.glr)
  d_opt = tf.train.AdamOptimizer(FLAGS.dlr)

  global_step = tf.train.get_or_create_global_step()

  global_step_update = global_step.assign_add(1)

  d_loss1, g_loss1, generated_genre2 = g1.loss(genre1_songs, genre2_songs)
  d_loss2, g_loss2, generated_genre1 = g2.loss(genre2_songs, genre1_songs)

  cycle1 = g2.generate(generated_genre2)
  cycle_loss1 = tf.cast(
      tf.losses.mean_squared_error(cycle1, genre1_songs), dtype=tf.float64)

  cycle2 = g1.generate(generated_genre1)
  cycle_loss2 = tf.cast(
      tf.losses.mean_squared_error(cycle2, genre2_songs), dtype=tf.float64)

  g_loss1 = g_loss1 + FLAGS.cycle_weight * cycle_loss1
  g_loss2 = g_loss2 + FLAGS.cycle_weight * cycle_loss2

  train_op = tf.group([
      g_opt.minimize(g_loss1, var_list=g1.gen.trainable_variables),
      g_opt.minimize(g_loss2, var_list=g2.gen.trainable_variables),
      d_opt.minimize(d_loss1, var_list=g1.disc.trainable_variables),
      d_opt.apply_gradients(d_loss2, g2.disc.trainable_variables)
  ])

  with tf.train.MonitoredTrainingSession() as sess:

    for _ in range(FLAGS.train_steps):
      _, DLoss1, GLoss1, DLoss2, GLoss2 = sess.run(
          [train_op, d_loss1, g_loss1, d_loss2, g_loss2])
      sess.run(global_step_update)
      print("Step {} : D loss : {} : G loss {}".format(
          global_step.eval(), DLoss1 + DLoss2, GLoss1 + GLoss2))


if __name__ == '__main__':
  flags.mark_flags_as_required(['genre1', 'genre2'])
  app.run(main)
