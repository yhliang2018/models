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
"""Definitions for DeepSpeech2 model.

Some abbreviations used in the code base:
NeuMF: Neural Matrix Factorization
NCF: Neural Collaborative Filtering
GMF: Generalized Matrix Factorization
MLP: Multi-Layer Perceptron

GMF applies a linear kernel to model the latent feature interactions, and MLP
uses a nonlinear kernel to learn the interaction function from data. NeuMF model
is a fused model of GMF and MLP to better model the complex user-item
interactions, and unifies the strengths of linearity of MF and non-linearity of
MLP for modeling the user-item latent structures.

In NeuMF model, it allows GMF and MLP to learn separate embeddings, and combine
the two models by concatenating their last hidden layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

_SUPPORTED_RNNS = {
    "lstm": tf.keras.layers.LSTM,
    "rnn": tf.keras.layers.SimpleRNN,
    "gru": tf.keras.layers.GRU,
}


def _sequence_wise(model, inputs):
  t, n = tf.shape(inputs)
  inputs = tf.reshape(inputs, [t*n, -1])
  inputs = model(inputs)
  inputs = tf.reshape(inputs, [t, n, -1])
  return inputs


def _conv_layer(cnn_input, filters, kernel_size, strides, layer_id):
  cnn_output = tf.keras.layers.Conv2D(
      filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
      activation="tanh", name="cnn_{}".format(layer_id))(cnn_input)
  output = tf.keras.layers.BatchNormalization(
      momentum=0.1, epsilon=1e-05)(cnn_output)
  return output


def _rnn_layer(input_data, rnn_cell, rnn_hidden_size, layer_id, rnn_activation,
              is_batch_norm, is_bidrectional):
  # Batch normalization
  if is_batch_norm:
    input_data = _sequence_wise(
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05),
        input_data)
  # RNN layer
  rnn_layer = rnn_cell(
      rnn_hidden_size, activation=rnn_activation, return_sequences=True,
      name="rnn_{}".format(layer_id))
  # Bidirectional rnn layer
  if is_bidirectional:
    rnn_layer = tf.keras.layers.Bidirectional(rnn_layer, merge_mode="sum")

  rnn_output = rnn_layer(input_data)

  return rnn_output


class DeepSpeech2(tf.keras.models.Model):
  """DeepSpeech 2 model."""

  def __init__(self, input_dim, num_rnn_layers, rnn_type, is_bidirectional,
               rnn_hidden_size, rnn_activation, num_classes, use_bias):
    """Initialize DeepSpeech2 model.

    Args:
      input_dim: an integer, the dimension of input dataset.
      num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.
      rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.
      is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
      rnn_hidden_size: an integer for the number of rnn units in each layer.
      num_classes: an integer, the number of output classes/labels.
      rnn_activation: a string to indicate rnn activation function. It can be
        one of tanh and relu.
    """
    # Input variables
    input_data = tf.keras.layers.Input(
        shape=(None, input_dim), name="the_input")

    # Two cnn layers
    conv_layer_1 = _conv_bn_layer(
        cnn_input, filters=32, kernel_size=(41, 11), strides=(2, 2),
        layer_id=1)
    conv_layer_2 = _conv_bn_layer(
        conv_layer_1, filters=32, kernel_size=(21, 11), strides=(2, 1),
        layer_id=2)

    # Five bidirectional GRU layers
    rnn_input = conv_layer_2
    rnn_cell = _SUPPORTED_RNNS[rnn_type]
    for layer in xrange(num_rnn_layers):
      is_batch_norm = (layer != 0)  # No batch normalization on the first layer
      rnn_input = _rnn_layer(rnn_input, rnn_cell, rnn_hidden_size, layer + 1,
                             rnn_activation, is_batch_norm, is_bidrectional)

    # FC layer
    fc_layer = tf.keras.models.Sequential(
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05),
        tf.keras.layers.Dense(
            num_classes, activation="softmax", use_bias=use_bias, name="logits")
    )
    fc_output = _sequence_wise(rnn_input, fc_layer)

    super(DeepSpeech2, self).__init__(inputs=[input_data], outputs=fc_output)
