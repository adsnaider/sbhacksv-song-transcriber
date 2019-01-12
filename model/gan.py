from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tensorflow.contrib import eager as tfe

import audio_models

tf.enable_eager_execution()

FLAGS = flags.FLAGS

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

  genre1 = tf.data.TFRecordDataset(FLAGS.genre1)
  genre1 = genre1.shuffle(buffer_size=100)
  genre1 = genre1.batch(FLAGS.batch_size)
  genre1 = genre1.repeat()

  genre2 = tf.data.TFRecordDataset(FLAGS.genre2)
  genre2 = genre2.shuffle(buffer_size=100)
  genre2 = genre2.batch(FLAGS.batch_size)
  genre2 = genre2.repeat()

  genre1 = tfe.Iterator(
      tf.data.Dataset.from_tensor_slices(
          np.random.normal(size=[100, 6000, 1])).batch(1))
  genre2 = tfe.Iterator(
      tf.data.Dataset.from_tensor_slices(
          np.random.normal(size=[100, 6000, 1])).batch(1))

  g1 = GAN(audio_models.AudioEncoderDecoder(), audio_models.AudioClassifier())
  g2 = GAN(audio_models.AudioEncoderDecoder(), audio_models.AudioClassifier())

  g_opt = tf.train.AdamOptimizer(FLAGS.glr)
  d_opt = tf.train.AdamOptimizer(FLAGS.dlr)

  global_step = tf.train.get_or_create_global_step()

  for i in range(FLAGS.train_steps):
    global_step.assign_add(1)
    genre1_songs = next(genre1)
    genre2_songs = next(genre2)

    with tf.GradientTape(persistent=True) as tape:
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

    g_opt.apply_gradients(
        zip(
            tape.gradient(g_loss1, g1.gen.trainable_variables),
            g1.gen.trainable_variables))
    g_opt.apply_gradients(
        zip(
            tape.gradient(g_loss2, g2.gen.trainable_variables),
            g2.gen.trainable_variables))
    d_opt.apply_gradients(
        zip(
            tape.gradient(d_loss1, g1.disc.trainable_variables),
            g1.disc.trainable_variables))
    d_opt.apply_gradients(
        zip(
            tape.gradient(d_loss2, g2.disc.trainable_variables),
            g2.disc.trainable_variables))

    del tape

    print("Step {} : D loss : {} : G loss {}".format(
        i, (d_loss1 + d_loss2).numpy(), (g_loss1 + g_loss2).numpy()))


if __name__ == '__main__':
  flags.mark_flags_as_required(['genre1', 'genre2'])
  app.run(main)
