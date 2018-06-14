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
from official.utils.export import export
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers

def evaluate_model(estimator, labels, targets, input_fn_eval):
  """Model evaluation with HR and NDCG metrics.

  The evaluation protocol is to rank the test interacted item (truth items)
  among the randomly chosen 100 items that are not interacted by the user.
  The performance of the ranked list is judged by Hit Ratio (HR) and Normalized
  Discounted Cumulative Gain (NDCG).

  For evaluation, the ranked list is truncated at 10 for both metrics. As such,
  the HR intuitively measures whether the test item is present on the top-10
  list, and the NDCG accounts for the position of the hit by assigning higher
  scores to hits at top ranks. Both metrics are calculated for each test user,
  and the average scores are reported.

  Args:
    estimator: The Estimator.
    batch_size: An integer, the batch size specified by user.
    num_gpus: An integer, the number of gpus specified by user.
    ncf_dataset: An NCFDataSet object, which contains the information about
      test/eval dataset, such as:
      eval_true_items, which is a list of test items (true items) for HR and
        NDCG calculation. Each item is for one user.
      eval_all_items, which is a nested list. Each entry is the 101 items
        (1 ground truth item and 100 negative items) for one user.

  Returns:
    eval_results: A dict of evaluation results for benchmark logging.
      eval_results = {
        _HR_KEY: hr,
        _NDCG_KEY: ndcg,
        tf.GraphKeys.GLOBAL_STEP: global_step
      }
      where hr is an integer indicating the average HR scores across all users,
      ndcg is an integer representing the average NDCG scores across all users,
      and global_step is the global step
  """
  # Get predictions
  predictions = estimator.predict(input_fn=input_fn_eval)
  predicted_logits = [p["logits"] for p in predictions]

  # (decoded_dense, log_prob)
  decoded_output = tf.keras.backend.ctc_decode(predicted_logits, seq_lens)

  decoder = DeepSpeechDecoder(labels)
  decoded_strings = decoder.decode(decoded_output[1])
  target_strings = decoder.decode(targets) # targets should be a list of numeric sequences

  # WER and CER
  total_wer, total_cer = 0, 0
  wer, cer = 0, 0
  for x in range(len(target_strings)):
    wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
    cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
  total_cer += cer
  total_wer += wer

  hits, ndcgs = [], []
  num_users = len(ncf_dataset.eval_true_items)
  # Reshape the predicted scores and each user takes one row
  predicted_scores_list = np.asarray(
      all_predicted_scores).reshape(num_users, -1)

  for i in range(num_users):
    items = ncf_dataset.eval_all_items[i]
    predicted_scores = predicted_scores_list[i]
    # Map item and score for each user
    map_item_score = {}
    for j, item in enumerate(items):
      score = predicted_scores[j]
      map_item_score[item] = score

    # Evaluate top rank list with HR and NDCG
    ranklist = heapq.nlargest(_TOP_K, map_item_score, key=map_item_score.get)
    true_item = ncf_dataset.eval_true_items[i]
    hr = _get_hr(ranklist, true_item)
    ndcg = _get_ndcg(ranklist, true_item)
    hits.append(hr)
    ndcgs.append(ndcg)

  # Get average HR and NDCG scores
  hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
  global_step = estimator.get_variable_value(tf.GraphKeys.GLOBAL_STEP)
  eval_results = {
      _HR_KEY: hr,
      _NDCG_KEY: ndcg,
      tf.GraphKeys.GLOBAL_STEP: global_step
  }
  return eval_results


def convert_keras_to_estimator(keras_model, num_gpus, labels, model_dir):
  """Configure and convert keras model to Estimator.

  Args:
    keras_model: A Keras model object.
    num_gpus: An integer, the number of gpus.
    model_dir: A string, the directory to save and restore checkpoints.

  Returns:
    est_model: The converted Estimator.
  """
  # Define CTC loss
  def _ctc_lambda_func(y_pred, labels, input_length, label_length):
    return tf.keras.backend.ctc_batch_cost(
        labels, y_pred, input_length, label_length)

  def _ctc_loss(y_true, y_pred):
    # Input of labels and other CTC requirements
    labels = Input(name="the_labels", shape=[None,], dtype="int32")
    input_length = Input(name="input_length", shape=[1], dtype="int32")
    label_length = Input(name="label_length", shape=[1], dtype="int32")

    # Keras doesn"t currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss = Lambda(_ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length])
    return loss

  optimizer = tf.keras.optimizers.SGD(
      lr=flags_obj.lr, momentum=flags_obj.momentum, decay=flags_obj.l2,
      nesterov=True)

  keras_model.compile(optimizer=optimizer, loss=_ctc_loss)

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

# Create deep speech model and convert it to Estimator
  tf.logging.info("Creating Estimator from Keras model...")
  with open(flags_obj.labels_path) as label_file:
    labels = str("".join(json.load(label_file)))
  num_classes = len(labels)

  keras_model = deep_speech_model.DeepSpeech2(
      input_dim, flags_obj.num_rnn_layers, flags_obj.rnn_type,
      flags_obj.is_bidirectional, flags_obj.rnn_hidden_size,
      flags_obj.rnn_activation, num_classes, flags_obj.use_bias)

  num_gpus = flags_core.get_num_gpus(FLAGS)
  estimator = convert_keras_to_estimator(
      keras_model, num_gpus, flags_obj.model_dir)

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
    return input_function(
        is_training=True, data_dir=flags_obj.data_dir,
        batch_size=per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=flags_obj.epochs_between_evals,
        num_gpus=flags_core.get_num_gpus(flags_obj))

  def input_fn_eval():
    return input_function(
        is_training=False, data_dir=flags_obj.data_dir,
        batch_size=per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=1)

  total_training_cycle = (flags_obj.train_epochs //
                          flags_obj.epochs_between_evals)
  for cycle_index in range(total_training_cycle):
    tf.logging.info("Starting a training cycle: %d/%d",
                    cycle_index, total_training_cycle)

    estimator.train(input_fn=input_fn_train, hooks=train_hooks)

    tf.logging.info("Starting to evaluate.")

    eval_results = evaluate_model(
        estimator, labels, input_fn=input_fn_eval)

    benchmark_logger.log_evaluation_result(eval_results)

    if model_helpers.past_stop_threshold(
        flags_obj.stop_threshold, eval_results["accuracy"]):
      break

  if flags_obj.export_dir is not None:
    # Exports a saved model for the given classifier.
    input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
        shape, batch_size=flags_obj.batch_size)
    classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn)


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
      model_dir="/tmp/deep_speech/model/",
      data_dir="/tmp/deep_speech/data/",
      export_dir="/tmp/deep_speech/saved_model/",
      train_epochs=10,
      batch_size=5,
      hooks="ProfilerHook")

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
