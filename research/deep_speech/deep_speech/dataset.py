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

import numpy as np
from absl import app as absl_app
from featurizer import AudioFeaturizer, TextFeaturizer
import scipy.io.wavfile as wavfile
import tensorflow as tf


class AudioConfig(object):
  """Config for audio."""

  def __init__(self,
               sample_rate,
               frame_length,
               frame_step,
               fft_length=None,
               spect_type="linear"):
    self.sample_rate = sample_rate
    self.frame_length = frame_length
    self.frame_step = frame_step
    self.fft_length = fft_length
    self.spect_type = spect_type


class DatasetConfig(object):
  """Config for the data set."""

  def __init__(self, audio_config, train_data_path, dev_data_path,
               test_data_path, vocab_file_path):
    """Initialize the configs for deep speech dataset.

    Args:
      audio_config:
      train_data_path:
      dev_data_path:
      test_data_path:
      vocab_file_path:
    Raises:
      RuntimeError: file path not exist.
    """
    self.audio_config = audio_config
    # assert tf.gfile.Exists(train_data_path)
    # assert tf.gfile.Exists(dev_data_path)
    assert tf.gfile.Exists(test_data_path)
    assert tf.gfile.Exists(vocab_file_path)
    self.train_data_path = train_data_path
    self.dev_data_path = dev_data_path
    self.test_data_path = test_data_path
    self.vocab_file_path = vocab_file_path


class DeepSpeechDataset(object):
  """Data provider for training/evaluation."""

  def __init__(self, dataset_config):
    """Initialize the dataset class.

    Each dataset file contains three columns: "wav_filename", "wav_filesize",
    and "transcript". This function parses the csv file and stores each example
    by the increasing order of audio length (indicated by wav_filesize).

    Args:
      dataset_config:
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

    # self.train_features, self.train_labels = self.load_data(
    #    self.config.train_data_path)
    self.test_features, self.test_labels = self.load_data(
        self.config.test_data_path)

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

  def load_data(self, file_path):
    """Generate a list of waveform, transcript pair.

    Note that the waveforms are ordered in increasing length, so that audio
    samples in a mini-batch have similar length.

    Args:
    Returns:
    """

    lines = tf.gfile.Open(file_path, "r").readlines()
    lines = [line.split("\t") for line in lines]
    lines.sort(key=lambda item: item[1])
    features = [self._preprocess_audio(line[0]) for line in lines]
    labels = [self._preprocess_transcript(line[2]) for line in lines]
    return features, labels


def input_fn(training, batch_size, deep_speech_dataset, repeat=1):
  """Input function for model training and evaluation.
  Args:
    training:
    batch_size:
    deep_speech_dataset:
    repeat:

  Returns:
    a tf.data.Dataset object for model to consume.
  """
  features = deep_speech_dataset.test_features
  features = np.expand_dims(features, axis=1)
  # print("deep_speech_dataset.test_features", features)
  # labels = tf.convert_to_tensor(deep_speech_dataset.test_labels)
  # labels = tf.expand_dims(labels, 0)
  labels = deep_speech_dataset.test_labels
  labels = np.expand_dims(labels, axis=1)
  # print("deep_speech_dataset.test_labels", labels)

  if training:
    dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels)
    )
    print("dataset", dataset)

  else:
    dataset = tf.data.Dataset.from_tensor_slices(
        (deep_speech_dataset.test_features, deep_speech_dataset.test_labels))
  # Repeat and batch the dataset
  dataset = dataset.repeat(repeat)
  dataset = dataset.batch(batch_size)

  # Prefetch to improve speed of input pipeline.
  dataset = dataset.prefetch(1)
  return dataset


def main(_):
  audio_conf = AudioConfig(16000, 25, 10)
  data_conf = DatasetConfig(audio_conf, "", "",
                            "test-clean.1.csv",
                            "vocabulary.txt")
  data_set = DeepSpeechDataset(data_conf)
  sess = tf.Session()
  for feature in data_set.test_features:
    print(feature)
    print(sess.run(feature))

  sess.close()
  print(data_set.test_labels[:10])



if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  absl_app.run(main)

