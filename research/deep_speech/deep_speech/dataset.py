#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""Generate tf.data.Dataset object for training/evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl import app as absl_app
from featurizer import AudioFeaturizer
from featurizer import TextFeaturizer
import numpy as np
import scipy.io.wavfile as wavfile
import tensorflow as tf


_SHUFFLE_BUFFER_SIZE = 1024

class AudioConfig(object):
  """Configs for spectrogram extraction from audio."""

  def __init__(self,
               sample_rate,
               frame_length,
               frame_step,
               fft_length=None,
               spect_type="linear"):
    """Initialize the AudioConfig class.

    Args:
      sample_rate: an integer denoting the sample rate of the input waveform.
      frame_length: an integer for the length of a spectrogram frame, in ms.
      frame_step: an integer for the frame stride, in ms.
      fft_length: an integer for the number of fft bins.
      spect_type: a string for the type of spectrogram to be extracted.
    """

    self.sample_rate = sample_rate
    self.frame_length = frame_length
    self.frame_step = frame_step
    self.fft_length = fft_length
    self.spect_type = spect_type


class DatasetConfig(object):
  """Config class for generating the DeepSpeechDataset."""

  def __init__(self, audio_config, data_path, vocab_file_path):
    """Initialize the configs for deep speech dataset.

    Args:
      audio_config: AudioConfig object specifying the audio-related configs.
      data_path: a string denoting the full path of a manifest file.
      vocab_file_path: a string denoting the vocabulary file path.

    Raises:
      RuntimeError: file path not exist.
    """

    self.audio_config = audio_config
    assert tf.gfile.Exists(data_path)
    assert tf.gfile.Exists(vocab_file_path)
    self.data_path = data_path
    self.vocab_file_path = vocab_file_path


class DeepSpeechDataset(object):
  """Dataset class for training/evaluation of DeepSpeech model."""

  def __init__(self, dataset_config):
    """Initialize the class.

    Each dataset file contains three columns: "wav_filename", "wav_filesize",
    and "transcript". This function parses the csv file and stores each example
    by the increasing order of audio length (indicated by wav_filesize).

    Args:
      dataset_config: DatasetConfig object.
    """
    self.config = dataset_config
    # Instantiate audio feature extractor.
    self.audio_featurizer = AudioFeaturizer(
        sample_rate=self.config.audio_config.sample_rate,
        frame_length=self.config.audio_config.frame_length,
        frame_step=self.config.audio_config.frame_step,
        fft_length=self.config.audio_config.fft_length,
        spect_type=self.config.audio_config.spect_type)
    # Instantiate text feature extractor.
    self.text_featurizer = TextFeaturizer(
        vocab_file=self.config.vocab_file_path)

    self.speech_labels = self.text_featurizer.labels

    self.features, self.labels = self._preprocess_data(self.config.data_path)

  def _preprocess_data(self, file_path):
    """Generate a list of waveform, transcript pair.

    Note that the waveforms are ordered in increasing length, so that audio
    samples in a mini-batch have similar length.
    """

    lines = tf.gfile.Open(file_path, "r").readlines()
    lines = [line.split("\t") for line in lines]
    lines.sort(key=lambda item: item[1])
    features = [self._preprocess_audio(line[0]) for line in lines]
    labels = [self._preprocess_transcript(line[2]) for line in lines]
    return features, labels

  def _preprocess_audio(self, audio_file_path):
    """Load the audio file in memory."""
    sample_rate, data = wavfile.read(audio_file_path)
    assert sample_rate == self.config.audio_config.sample_rate
    if data.dtype not in [np.float32, np.float64]:
      data = data.astype(np.float32) / np.iinfo(data.dtype).max
    feature = self.audio_featurizer.featurize(data)
    return (tf.Session().run(feature))  # return a numpy array rather than a tensor

  def _preprocess_transcript(self, transcript):
    return self.text_featurizer.featurize(transcript)


def input_fn(training, batch_size, deep_speech_dataset, repeat=1):
  """Input function for model training and evaluation.

  Args:
    training: a boolean value indicating whether we are in the training cycle.
    batch_size: an integer denoting the size of a batch.
    deep_speech_dataset: DeepSpeechDataset object.
    repeat: an integer for how many times to repeat the dataset.

  Returns:
    a tf.data.Dataset object for model to consume.
  """
  features = deep_speech_dataset.features
  labels = deep_speech_dataset.labels
  # for i in range(len(features)):
  #     feature = np.expand_dims(features[i], axis=2)
  #     # feature = features[i]
  #     print("features", feature.shape)
  #     print("labels", labels[i])
  #     print("input_length", feature.shape[0])
  #     print("label_length", len(labels[i]))


  def _data_gen():
    for i in range(len(features)):
      feature = np.expand_dims(features[i], axis=2)
      # input_length = np.expand_dims(feature.shape[0], axis=1)
      # label_length = np.expand_dims(len(labels[i]), axis=1)
      # feature = features[i]
      input_length = [199]
      label_length = [len(labels[i])]
      yield ({
          "features": feature,
          "labels": labels[i],
          "input_length": input_length,  # A list with one number
          "label_length": label_length
      })

  dataset = tf.data.Dataset.from_generator(
      generator=_data_gen,
      output_types = {
          "features": tf.float32,
          "labels": tf.int32,
          "input_length": tf.int32,
          "label_length": tf.int32
      })
  dataset = dataset.padded_batch(
      batch_size=batch_size,
      padded_shapes={
          "features": (tf.Dimension(None), 257, 1),
          "labels": tf.Dimension(None),
          "input_length": tf.Dimension(None),
          "label_length": tf.Dimension(None)
      })
  print("after padding dataset", dataset)
  """
  if training:
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE)

  dataset = dataset.padded_batch(
      batch_size=batch_size,
      padded_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None])),
  )

  g = tf.Graph()
  with tf.Session(graph=g).as_default() as sess, g.as_default():
    ds_it = dataset.make_one_shot_iterator()
    row = ds_it.get_next()
    for i in range(int(len(features)//2)):
      x = sess.run(row)
      # print(x)
      if i % 10 == 0:
        print(x)
    print("done")

  print("after padding dataset", dataset)
  """
  # Repeat and batch the dataset
  dataset = dataset.repeat(repeat)
  dataset = dataset.prefetch(1)

  print("before return", dataset.output_shapes)

  return dataset


def main(_):
  audio_conf = AudioConfig(16000, 25, 10)
  data_conf = DatasetConfig(
      audio_conf,
      "test-clean.2.csv",
      "vocabulary.txt"
  )
  data_set = DeepSpeechDataset(data_conf)
  # dataset_input_fn = input_fn(True, 2, data_set)
  # print("dataset from input_fn: ", dataset_input_fn)

  data = data_set.features
  labels = data_set.labels
  # print("feature length", len(data))
  # print("label length", len(labels))

  for i in range(len(data)):
      feature = np.expand_dims(data[i], axis=2)
      print("feature", feature.shape)
      # print("features", feature.shape)
      # print("labels", labels[i])
      input_length = feature.shape[0]
      label_length = len(labels[i])
      # print("input_length", input_length)
      # print("label_length", label_length)


  def _data_gen():
    for i in range(len(data)):
      feature = np.expand_dims(data[i], axis=2)
      # feature = data[i]
      # input_length = np.expand_dims(feature.shape[0], axis=1)
      # label_length = np.expand_dims(len(labels[i]), axis=1)
      input_length = [199]
      label_length = [len(labels[i])]
      print("input_length", input_length)
      print("label_length", label_length)
      yield ({
          "features": feature,
          "labels": labels[i],
          "input_length": input_length,
          "label_length": label_length
      })


  print("In Session:")
  g = tf.Graph()
  with tf.Session(graph=g).as_default() as sess, g.as_default():
    batch_size = 2
    dataset = tf.data.Dataset.from_generator(generator=_data_gen, output_types = {
            "features": tf.float32,
            "labels": tf.int32,
            "input_length": tf.int32,
            "label_length": tf.int32
        })
    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes={
            "features": (tf.Dimension(None), 257, 1),
            "labels": tf.Dimension(None),
            "input_length": tf.Dimension(None),
            "label_length": tf.Dimension(None)
        })
       # tf.Dimension(None), tf.Dimension(None)))
    # dataset = dataset.repeat(1)
    # dataset = dataset.prefetch(1)
    print("before return", dataset.output_shapes)

    ds_it = dataset.make_one_shot_iterator()
    print("ds_it", ds_it)
    row = ds_it.get_next()
    print("row", row)
    for i in range(math.ceil(len(data)/batch_size)):
      x = sess.run(row)
      print(x)
      # x = sess.run(row["input_length"])
      # print("input_length", x)
      # x = sess.run(row["label_length"])
      # print("label_length", x)
      # if i % 10 == 0:
      print(i, x)
    print("done")



  # print(len(data_set.features))
  # print(len(data_set.labels[0]))

  # sess = tf.Session()
  # for feature in data_set.features:
  #   print(sess.run(feature))
  # for label in data_set.labels:
  #   print(sess.run(label))
  # sess.close()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  absl_app.run(main)

