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
"""Deep Speech
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

# pylint: disable=g-bad-import-order
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.deep_speech import deep_speech_model
from official.deep_speech import dataset
from official.utils.export import export
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers

def evaluate_model(estimator, labels, targets, input_fn_eval):

  # Get predictions
  predictions = estimator.predict(input_fn=input_fn_eval, yield_single_examples=False)

  y_preds = []
  input_lengths= []
  for p in predictions:
    y_pred = p["y_pred"]
    y_preds.append(y_pred)
    input_length = p["ctc_input_length"]
    input_lengths.append(input_length)

  for i in range(len(y_preds)):
    y_pred_tensor = tf.convert_to_tensor(y_preds[i])
    # input_length_tensor = tf.convert_to_tensor(input_lengths[i])
    input_length_tensor = tf.squeeze(input_lengths[i],axis=1)
    print("i: ", i)
    print("y_pred_tensor", y_pred_tensor)
    print("input_length_tensor", input_length_tensor)

    decoded_output = tf.keras.backend.ctc_decode(y_pred_tensor, input_length_tensor)
    print("decoded output")
    print(decoded_output)



def convert_keras_to_estimator(keras_model, num_gpus, model_dir):
  """Configure and convert keras model to Estimator.

  Args:
    keras_model: A Keras model object.
    num_gpus: An integer, the number of gpus.
    model_dir: A string, the directory to save and restore checkpoints.

  Returns:
    est_model: The converted Estimator.
  """
  optimizer = tf.keras.optimizers.SGD(
      lr=flags_obj.learning_rate, momentum=flags_obj.momentum, decay=flags_obj.l2,
      nesterov=True)
  # print(optimizer)

  keras_model.compile(optimizer=optimizer, loss={"ctc": lambda y_true, y_pred: y_pred})
  print("keras model: ", keras_model)

  if num_gpus == 0:
    distribution = tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    distribution = tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)

  run_config = tf.estimator.RunConfig(train_distribute=distribution)

  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model, model_dir=model_dir, config=run_config)

  return estimator


def per_device_batch_size(batch_size, num_gpus):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.

  Note that this should eventually be handled by DistributionStrategies
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.

  Args:
    batch_size: Global batch size to be divided among devices. This should be
      equal to num_gpus times the single-GPU batch_size for multi-gpu training.
    num_gpus: How many GPUs are used with DistributionStrategies.

  Returns:
    Batch size per device.

  Raises:
    ValueError: if batch_size is not divisible by number of devices
  """
  if num_gpus <= 1:
    return batch_size

  remainder = batch_size % num_gpus
  if remainder:
    err = ("When running with multiple GPUs, batch size "
           "must be a multiple of the number of available GPUs. Found {} "
           "GPUs with a batch size of {}; try --batch_size={} instead."
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)
  return int(batch_size / num_gpus)


def main(_):
  with logger.benchmark_context(flags_obj):
    run_deep_speech(flags_obj)


def run_deep_speech(_):
  """Run deep speech training and eval loop."""
  # Data preprocessing
  # The file name of training and test dataset
  tf.logging.info("Data preprocessing...")
  sample_rate = 16000
  frame_length = 25
  frame_step = 10
  audio_conf = dataset.AudioConfig(sample_rate, frame_length, frame_step)
  data_conf = dataset.DatasetConfig(
      audio_conf,
      "test-clean.2.csv",
      "vocabulary.txt"
  )
  data_set = dataset.DeepSpeechDataset(data_conf)
  # feat_len = len(data_set.test_features)
  print("features", data_set.features[0].shape)
  print("labels", len(data_set.labels[0]))

# Create deep speech model and convert it to Estimator
  tf.logging.info("Creating Estimator from Keras model...")
  num_classes = len(data_set.speech_labels)
  print("speech_labels", data_set.speech_labels)

  input_shape = (875, 257, 1)

  keras_model = deep_speech_model.DeepSpeech2(
      input_shape, flags_obj.rnn_hidden_layers, flags_obj.rnn_type,
      flags_obj.is_bidirectional, flags_obj.rnn_hidden_size,
      flags_obj.rnn_activation, num_classes, flags_obj.use_bias)

  print(keras_model.summary(line_length=100))

  num_gpus = flags_core.get_num_gpus(flags_obj)
  estimator = convert_keras_to_estimator(
      keras_model, num_gpus, flags_obj.model_dir)
  print("estimator", estimator)

  # Benchmark logging
  run_params = {
      "batch_size": flags_obj.batch_size,
      "train_epochs": flags_obj.train_epochs,
      "rnn_hidden_size": flags_obj.rnn_hidden_size,
      "rnn_hidden_layers": flags_obj.rnn_hidden_layers,
      "rnn_activation": flags_obj.rnn_activation,
      "rnn_type": flags_obj.rnn_type,
      "is_bidirectional":flags_obj.is_bidirectional,
      "use_bias": flags_obj.use_bias
  }

  dataset_name = "LibriSpeech"
  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info("deep_speech", dataset_name, run_params,
                                test_id=flags_obj.benchmark_test_id)

  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      batch_size=flags_obj.batch_size)

  def input_fn_train():
    return dataset.input_fn(
        True, 1,
        data_set)

  def input_fn_eval():
    return dataset.input_fn(
        True, per_device_batch_size(flags_obj.batch_size, num_gpus),
        data_set)

  total_training_cycle = (flags_obj.train_epochs //
                          flags_obj.epochs_between_evals)
  for cycle_index in range(total_training_cycle):
    tf.logging.info("Starting a training cycle: %d/%d",
                    cycle_index + 1, total_training_cycle)

    estimator.train(input_fn=input_fn_train, hooks=train_hooks)

    # tf.logging.info("Starting to evaluate.")

    # eval_results = evaluate_model(
    #     estimator, labels, input_fn=input_fn_eval)

    benchmark_logger.log_evaluation_result(eval_results)

    if model_helpers.past_stop_threshold(
        flags_obj.stop_threshold, eval_results["accuracy"]):
      break

  if flags_obj.export_dir is not None:
    # Exports a saved model for the given classifier.
    input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
        shape, batch_size=flags_obj.batch_size)
    classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn)

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()

def define_deep_speech_flags():
  """Add flags for run_deep_speech."""
  # Add common flags
  flags_core.define_base()
  flags_core.define_performance(
      num_parallel_calls=False,
      inter_op=False,
      intra_op=False,
      synthetic_data=False,
      max_train_steps=False,
      dtype=False
  )
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags_core.set_defaults(
      model_dir="/tmp/deep_speech_model/",
      data_dir="/tmp/deep_speech_data/",
      export_dir="/tmp/deep_speech_saved_model/",
      train_epochs=10,
      batch_size=1,
      hooks="")

  # Add deep_speech-specific flags
  # RNN related flags
  flags.DEFINE_integer(
      name="rnn_hidden_size", default=800,
      help=flags_core.help_wrap("The hidden size of RNNs."))

  flags.DEFINE_integer(
      name="rnn_hidden_layers", default=5,
      help=flags_core.help_wrap("The number of RNN layers."))

  flags.DEFINE_bool(
      name="use_bias", default=True,
      help=flags_core.help_wrap(
          "Use bias"))

  flags.DEFINE_bool(
      name="is_bidirectional", default=True,
      help=flags_core.help_wrap(
          "If rnn unit is bidirectional"))

  flags.DEFINE_enum(
      name="rnn_type", default="gru",
      enum_values=["gru", "rnn", "lstm"], case_sensitive=False,
      help=flags_core.help_wrap(
          "Type of RNN cell."))

  flags.DEFINE_enum(
      name="rnn_activation", default="tanh",
      enum_values=["tanh", "relu"], case_sensitive=False,
      help=flags_core.help_wrap(
          "Type of the activation within RNN."))

  # Training related flags
  flags.DEFINE_float(
      name="learning_rate", default=0.0003,
      help=flags_core.help_wrap("The initial learning rate."))

  flags.DEFINE_float(
      name="learning_anneal", default=1.1,
      help=flags_core.help_wrap(
          "Annealing applied to learning rate every epoch."))

  flags.DEFINE_float(
      name="momentum", default=0.9,
      help=flags_core.help_wrap("Momentum to accelerate SGD optimizer."))

  flags.DEFINE_float(
      name="l2", default=0,
      help=flags_core.help_wrap("L2 penalty (decay of SGD optimizer)."))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_deep_speech_flags()
  flags_obj = flags.FLAGS
  absl_app.run(main)
