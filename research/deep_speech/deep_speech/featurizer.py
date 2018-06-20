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
"""Utility class for extracting features from the text and audio input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import codecs
import numpy as np
import tensorflow as tf


class AudioFeaturizer(object):
  """Class to extract spectrogram features from the audio input."""

  def __init__(self,
               sample_rate=16000,
               frame_length=25,
               frame_step=10,
               fft_length=None,
               window_fn=functools.partial(tf.contrib.signal.hann_window, periodic=True),
               spect_type="linear"):
    """Initialize the audio featurizer class with the audio configs.

    Args:
      sample_rate: an integer denoting the sample rate of the input waveform.
      frame_length: an integer for the length of a spectrogram frame, in ms.
      frame_step: an integer for the frame stride, in ms.
      fft_length: an integer for the number of fft bins.
      window_fn: windowing function.
      spect_type: a string for the type of spectrogram to be extracted.
      Currently only support 'linear', otherwise will raise a value error.

    Raises:
      ValueError: In case of invalid arguments for `spect_type`.
    """
    if spect_type != "linear":
      raise ValueError("Unsupported spectrogram type: %s" % spect_type)
    self.sample_rate = sample_rate
    self.frame_length = frame_length
    self.frame_step = frame_step
    self.fft_length = fft_length
    self.window_fn = window_fn
    self.spect_type = spect_type
    self.adjusted_frame_length = int(self.sample_rate * self.frame_length / 1e3)
    self.adjusted_frame_step = int(self.sample_rate * self.frame_step / 1e3)
    self.adjusted_fft_length = fft_length if fft_length else int(2**(np.ceil(
        np.log2(self.adjusted_frame_length))))

  def featurize(self, waveforms):
    """Extract spectrogram feature tensors from the waveforms."""
    return self._compute_linear_spectrogram(waveforms)

  def _compute_linear_spectrogram(self, waveforms):
    """Compute the linear-scale, magnitude spectrograms for the input waveforms.

    Args:
      waveforms: float32 tensor with shape [batch_size, max_len]
    Returns:
      a float 32 tensor with shape [batch_size, len, num_bins]
    """

    # `stfts` is a complex64 Tensor representing the Short-time Fourier
    # Transform of each signal in `signals`. Its shape is
    # [batch_size, ?, fft_unique_bins] where fft_unique_bins =
    # fft_length // 2 + 1.
    stfts = tf.contrib.signal.stft(
        waveforms,
        frame_length=self.adjusted_frame_length,
        frame_step=self.adjusted_frame_step,
        fft_length=self.adjusted_fft_length,
        window_fn=self.window_fn,
        pad_end=True)

    # An energy spectrogram is the magnitude of the complex-valued STFT.
    # A float32 Tensor of shape [batch_size, ?, 513].
    magnitude_spectrograms = tf.abs(stfts)
    return magnitude_spectrograms

  def _compute_mel_filterbank_features(self, waveforms):
    """Compute the mel filterbank features."""
    raise NotImplementedError("MFCC feature extraction not supported yet.")


class TextFeaturizer(object):
  """Extract text feature based on char-level granularity. By looking up the
  vocabulary table, each input string (one line of transcript) will be converted
  to a sequence of integer indexes.
  """

  def __init__(self, vocab_file):
    lines = []
    with codecs.open(vocab_file, "r", "utf-8") as fin:
      lines.extend(fin.readlines())
    self.token_to_idx = {}
    self.idx_to_token = {}
    self.labels = ""
    idx = 0
    for line in lines:
      if line.startswith("#"):
        # Skip reading comment line.
        continue
      line = line[:-1]  # Strip the new line.
      self.token_to_idx[line] = idx
      self.idx_to_token[idx] = line
      self.labels += line
      idx += 1

  def featurize(self, text):
    """Convert string to a list of integers."""
    tokens = list(text.strip().lower())
    return [self.token_to_idx[token] for token in tokens]

  def revert_featurize(self, feat):
    """Convert a list of integers to a string."""
    return "".join(self.idx_to_token[idx] for idx in feat)

