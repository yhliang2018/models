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
"""Network structure for DeepSpeech model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Supported rnn cells
SUPPORTED_RNNS = {
    "lstm": tf.keras.layers.LSTM,
    "rnn": tf.keras.layers.SimpleRNN,
    "gru": tf.keras.layers.GRU,
}

# Parameters for batch normalization
_MOMENTUM = 0.1
_EPSILON = 1e-05


def _conv_bn_layer(model, zero_padding, filters, kernel_size, strides, layer_id):
  """2D convolution + batch normalization layer.

  Args:
    cnn_input: input data for convolution layer.
    filters: an integer, number of output filters in the convolution.
    kernel_size: a tuple specifying the height and width of the 2D convolution
      window.
    strides: a tuple specifying the stride length of the convolution.
    layer_id: an integer specifying the layer index.

  Returns:
    tensor output from the current layer.
  """
  model.add(tf.keras.layers.ZeroPadding2D(padding=zero_padding))
  model.add(tf.keras.layers.Conv2D(
      filters=filters, kernel_size=kernel_size, strides=strides, padding="valid",
      activation="linear", use_bias=False,
      name="cnn_{}".format(layer_id)))
  model.add(tf.keras.layers.BatchNormalization(
      momentum=_MOMENTUM, epsilon=_EPSILON))
  return model


def _rnn_layer(model, rnn_cell, rnn_hidden_size, layer_id, rnn_activation,
               is_batch_norm, is_bidirectional):
  """Defines a batch normalization + rnn layer.

  Args:
    input_data: input tensors for the current layer.
    rnn_cell: RNN cell instance to use.
    rnn_hidden_size: an integer for the dimensionality of the rnn output space.
    layer_id: an integer for the index of current layer.
    rnn_activation: activation function to use.
    is_batch_norm: a boolean specifying whether to perform batch normalization
      on input states.
    is_bidirectional: a boolean specifying whether the rnn layer is
      bi-directional.

  Returns:
    tensor output for the current layer.
  """
  if is_batch_norm:
    model.add(tf.keras.layers.BatchNormalization(
        momentum=_MOMENTUM, epsilon=_EPSILON))
  rnn_layer = rnn_cell(
      rnn_hidden_size, activation=rnn_activation, return_sequences=True,
      use_bias=False, name="rnn_{}".format(layer_id))
  if is_bidirectional:
    rnn_layer = tf.keras.layers.Bidirectional(rnn_layer, merge_mode="sum")
  model.add(rnn_layer)
  return model


def create_model(num_rnn_layers, rnn_type, is_bidirectional,
                 rnn_hidden_size, rnn_activation, num_classes, use_bias):
  """Create DeepSpeech model.

  Args:
    input_shape: an tuple to indicate the dimension of input dataset. It has
      the format of [time_steps(T), feature_bins(F), channel(1)]
    num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.
    rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.
    is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
    rnn_hidden_size: an integer for the number of hidden states in each unit.
    rnn_activation: a string to indicate rnn activation function. It can be
      one of tanh and relu.
    num_classes: an integer, the number of output classes/labels.
    use_bias: a boolean specifying whether to use bias in the last fc layer.

  Returns:
  A tf.keras.Model.
  """
  model = tf.keras.Sequential()
  # Two cnn layers
  # Perform zero padding to amend the long sequences
  model = _conv_bn_layer(
      model, zero_padding=(20, 5), filters=32, kernel_size=(41, 11),
      strides=(2, 2), layer_id=1)

  model = _conv_bn_layer(
      model, zero_padding=(10, 5), filters=32, kernel_size=(21, 11),
      strides=(2, 1), layer_id=2)
  # output of conv_layer2 with the shape of
  # [batch_size (N), times (T), features (F), channels (C)]

  # RNN layers.
  # Convert the conv output to rnn input
  model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

  rnn_cell = SUPPORTED_RNNS[rnn_type]
  for layer_counter in xrange(num_rnn_layers):
    # No batch normalization on the first layer
    is_batch_norm = (layer_counter != 0)
    model = _rnn_layer(
        model, rnn_cell, rnn_hidden_size, layer_counter + 1,
        rnn_activation, is_batch_norm, is_bidirectional)

  # FC layer with batch norm
  model.add(tf.keras.layers.BatchNormalization(
      momentum=_MOMENTUM, epsilon=_EPSILON))

  model.add(tf.keras.layers.Dense(
      num_classes, activation="softmax", use_bias=use_bias))



  return model
