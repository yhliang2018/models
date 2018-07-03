# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deep speech decoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from nltk.metrics import distance
import numpy as np
from six.moves import xrange
import tensorflow as tf


class DeepSpeechDecoder(object):
  """Basic decoder class from which all other decoders inherit.

  Implements several helper functions. Subclasses should implement the decode()
  method.
  """

  def __init__(self, labels):
    """Decoder initialization.

    Arguments:
      labels (string): mapping from integers to characters.
      blank_index (int, optional): index for the blank '_' character.
        Defaults to 0.
      space_index (int, optional): index for the space ' ' character.
        Defaults to 28.
    """
    # e.g. labels = "[a-z]' _"
    self.labels = labels
    self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])

  def convert_to_string(self, sequence):
    """Convert a sequence of indexes into corresponding string."""
    return ''.join([self.int_to_char[i] for i in sequence])

  def wer(self, output, target):
    """Computes the Word Error Rate (WER).

    WER is defined as the edit distance between the two provided sentences after
    tokenizing to words.

    Args:
      output: string of the decoded output.
      target: a string for the true label.

    Returns:
      A float number for the WER of the current sentence pair.
    """
    # Map each word to a new char.
    words = set(output.split() + target.split())
    word2char = dict(zip(words, range(len(words))))

    new_output = [chr(word2char[w]) for w in output.split()]
    new_target = [chr(word2char[w]) for w in target.split()]

    return distance.edit_distance(''.join(new_output), ''.join(new_target))

  def cer(self, output, target):
    """Computes the Character Error Rate (CER).

    CER is  defined as the edit distance between the given strings.

    Args:
      output: a string of the decoded output.
      target: a string for the ground truth label.

    Returns:
      A float number denoting the CER for the current sentence pair.
    """
    return distance.edit_distance(output, target)

  def decode(self, logits):
    best = list(np.argmax(logits, axis=1))
    merge = [k for k,g in itertools.groupby(best)]
    return self.convert_to_string(merge)
