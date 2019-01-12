from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from scipy.io.wavfile import read as read_wav

logging.set_verbosity(tf.logging.DEBUG)
tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'Path to directory of waves')
flags.DEFINE_string('tracks_metadata', None, "Track metadata csv file.")
flags.DEFINE_string('output_dir', None, 'Path to store tfrecords')


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_records(sorted_audio, out_dir):
  for i, genre in enumerate(sorted_audio):
    record_file = os.path.join(out_dir, str(genre) + '.tfrecords')
    if not tf.gfile.Exists(record_file):
      writer = tf.python_io.TFRecordWriter(record_file)
      logging.debug('Processesing Genre: {}  :   {}/{}'.format(
          genre, i, len(sorted_audio)))

      for s, track in enumerate(sorted_audio[genre]):
        if s % 30 == 0:
          logging.debug("Processed {}/{} tracks in {}".format(
              s, len(sorted_audio[genre]), genre))
        track = os.path.join(FLAGS.input_dir, track)
        with tf.gfile.Open(track, 'rb') as f:
          sample_rate, song = read_wav(f)

        # Make sure we have a channel axis.
        if len(song.shape) == 1:
          song = np.expand_dims(song, -1)

        # Make mono
        song = np.mean(song, axis=1)

        feature = {
            'song': bytes_feature(tf.compat.as_bytes(song.tostring())),
            'sample_rate': int64_feature(sample_rate)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
      writer.close()
    else:
      logging.info("{} Already exists. Skipping.".format(record_file))


def main(argv):
  del argv
  if not os.path.exists('genres.pickle'):
    logging.info("Reading metadata")
    with tf.gfile.Open(FLAGS.tracks_metadata, 'rb') as f:
      track_metadata = pd.read_csv(f, index_col=0, header=[0, 1])

    logging.info("Getting wave files.")
    waves = tf.gfile.ListDirectory(FLAGS.input_dir)

    logging.info("Processing wave files.")
    sorted_genres = {}
    for wave in waves:
      filename = os.path.splitext(os.path.basename(wave))[0]
      genres = eval(track_metadata.loc[int(filename)]['track']['genres'])

      for genre in genres:
        if not genre in sorted_genres:
          sorted_genres[genre] = []
        sorted_genres[genre].append(wave)

      logging.debug("Done with track: {}".format(wave))
    with open("genres.pickle", 'wb') as f:
      pickle.dump(sorted_genres, f, protocol=pickle.HIGHEST_PROTOCOL)
  else:
    logging.debug("Found pickle file. Skipping.")
    with open('genres.pickle', 'rb') as f:
      sorted_genres = pickle.load(f)

  logging.info("Done sorting tracks. Writing tfrecords now.")
  write_records(sorted_genres, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
