from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

print("TF VERSION: !!!!!! ", tf.__version__)


class Conv1DTranspose(tf.keras.layers.Conv2DTranspose):

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Conv1DTranspose, self).__init__(
        filters, (kernel_size, 1), (strides, 1),
        data_format='channels_last',
        padding=padding,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

  def build(self, input_shape):
    super(Conv1DTranspose, self).build(input_shape)

  def call(self, inputs):
    return tf.squeeze(super(Conv1DTranspose, self).call(inputs), -2)

  def __call__(self, inputs):
    return super(Conv1DTranspose, self).__call__(tf.expand_dims(inputs, -2))

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    # Shape should be (Width, 1, channels)
    input_shape.insert(-1, 1)
    input_shape = tf.TensorShape(input_shape)
    output_shape = super(Conv1DTranspose,
                         self).compute_output_shape(input_shape).as_list()
    output_shape.pop(-2)
    return tf.TensorShape(output_shape)


# TODO(adam): Should split audio in chunks for training more easily.
# TODO(adam): Maybe use spectrogram as input instead.
# TODO(adam): Pad input to get correct decoder step shape.
class AudioEncoderDecoder(tf.keras.models.Model):

  def __init__(self, name='AudioEncoderDecoder', tracks=1):
    super(AudioEncoderDecoder, self).__init__(name=name)

    with tf.name_scope("encoder"):
      self.econv1 = tf.keras.layers.Conv1D(
          filters=3,
          kernel_size=10,
          padding='same',
          activation=tf.nn.leaky_relu)
      self.epool1 = tf.keras.layers.AveragePooling1D(
          pool_size=10, padding='same')
      self.enorm1 = tf.keras.layers.BatchNormalization()

      self.econv2 = tf.keras.layers.Conv1D(
          filters=5, kernel_size=8, padding='same', activation=tf.nn.leaky_relu)
      self.epool2 = tf.keras.layers.AveragePooling1D(
          pool_size=10, padding='same')
      self.enorm2 = tf.keras.layers.BatchNormalization()

      self.econv3 = tf.keras.layers.Conv1D(
          filters=10,
          kernel_size=5,
          padding='same',
          activation=tf.nn.leaky_relu)
      self.epool3 = tf.keras.layers.AveragePooling1D(
          pool_size=10, padding='same')
      self.enorm3 = tf.keras.layers.BatchNormalization()

    with tf.name_scope("decoder"):
      self.dconv1_transpose = Conv1DTranspose(
          filters=10,
          kernel_size=5,
          strides=10,
          padding='same',
          activation=tf.nn.leaky_relu)
      self.dnorm1 = tf.keras.layers.BatchNormalization()

      self.dconv2_transpose = Conv1DTranspose(
          filters=5,
          kernel_size=8,
          strides=10,
          padding='same',
          activation=tf.nn.leaky_relu)
      self.dnorm2 = tf.keras.layers.BatchNormalization()

      self.dconv3_transpose = Conv1DTranspose(
          filters=3,
          kernel_size=10,
          strides=10,
          padding='same',
          activation=tf.nn.leaky_relu)
      self.dnorm3 = tf.keras.layers.BatchNormalization()

    with tf.name_scope('output'):

      self.out_conv = tf.keras.layers.Conv1D(
          filters=tracks, kernel_size=10, padding='same', activation=None)

  def call(self, inputs):
    net = inputs
    net = l0 = inputs
    net = self.econv1(net)
    net = self.epool1(net)
    net = l1 = self.enorm1(net)
    net = self.econv2(net)
    net = self.epool2(net)
    net = l2 = self.enorm2(net)
    net = self.econv3(net)
    net = self.epool3(net)
    net = l3 = self.enorm3(net)

    net = self.dconv1_transpose(tf.concat([net, l3], axis=-1))
    net = self.dnorm1(net)
    net = self.dconv2_transpose(tf.concat([net, l2], axis=-1))
    net = self.dnorm2(net)
    net = self.dconv3_transpose(tf.concat([net, l1], axis=-1))
    net = self.dnorm3(net)

    net = self.out_conv(tf.concat([net, l0], axis=-1))

    return net


# TODO(adam): Should split audio in chunks for training more easily.
# TODO(adam): Maybe use spectrogram as input instead.
class AudioClassifier(tf.keras.models.Model):

  def __init__(self, name='AudioClassifier'):
    super(AudioClassifier, self).__init__(name=name)

    self.conv1 = tf.keras.layers.Conv1D(
        filters=3, kernel_size=10, padding='same', activation=tf.nn.leaky_relu)
    self.pool1 = tf.keras.layers.AveragePooling1D(pool_size=10, padding='same')
    self.norm1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv1D(
        filters=5, kernel_size=8, padding='same', activation=tf.nn.leaky_relu)
    self.pool2 = tf.keras.layers.AveragePooling1D(pool_size=10, padding='same')
    self.norm2 = tf.keras.layers.BatchNormalization()

    self.conv3 = tf.keras.layers.Conv1D(
        filters=10, kernel_size=5, padding='same', activation=tf.nn.leaky_relu)
    self.pool3 = tf.keras.layers.AveragePooling1D(pool_size=10, padding='same')
    self.norm3 = tf.keras.layers.BatchNormalization()

    with tf.name_scope("output"):
      self.fc = tf.keras.layers.Dense(units=1)

  def call(self, inputs):
    net = inputs
    net = self.conv1(net)
    net = self.pool1(net)
    net = self.norm1(net)
    net = self.conv2(net)
    net = self.pool2(net)
    net = self.norm2(net)
    net = self.conv3(net)
    net = self.pool3(net)
    net = self.norm3(net)

    net = tf.reduce_max(net, axis=1)
    net = self.fc(net)

    return net


# Simple main program to test the model works.
if __name__ == '__main__':
  tf.enable_eager_execution()

  import numpy as np

  input_data = np.random.normal(size=[20, 6000, 1])
  encoder_decoder = AudioEncoderDecoder(tracks=1)

  print(input_data)
  print(input_data.shape)
  out = encoder_decoder(tf.constant(input_data, dtype=tf.float32)).numpy()
  print(out)
  print(out.shape)

  classifier = AudioClassifier()
  out = classifier(tf.constant(input_data, dtype=tf.float32)).numpy()

  print(out)
  print(out.shape)
