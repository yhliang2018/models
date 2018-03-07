# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import multiprocessing

import tensorflow as tf


AUTO_STRING = "automatic"


class NearlyRawTextHelpFormatter(argparse.HelpFormatter):
  """Formatter for unified arg parser.

    This formatter allows explicit newlines and indentation but handles wrapping
  text where appropriate.
  """

  def _split_lines(self, text, width):
    output = []
    for line in text.splitlines():
      output.extend(self._split_for_length(line, width=width))
    return output

  @staticmethod
  def _split_for_length(text, width):
    out_lines = [[]]
    segments = [i + " " for i in text.split(" ")]
    segments[-1] = segments[-1][:-1]
    current_len = 0
    for segment in segments:
      if not current_len or current_len + len(segment) <= width:
        current_len += len(segment)
        out_lines[-1].append(segment)
      else:
        current_len = 0
        out_lines.append([segment])
    return ["".join(i) for i in out_lines]


class BaseParser(argparse.ArgumentParser):
  """Parent parser class for official models.

    This class is intended to house convenience functions, enforcement code, and
  universal or nearly universal arguments.
  """

  def __init__(self):
    super(BaseParser, self).__init__(
        formatter_class=NearlyRawTextHelpFormatter,
        allow_abbrev=False,  # abbreviations are handled explicitly.
    )
    self.post_parse_functions = []

  def parse_args(self, args=None, namespace=None):
    namespace = super(BaseParser, self).parse_args(args=args,
                                                   namespace=namespace)

    # Allow correctness checks, dynamic defaults, and other custom cleanup code.
    for cleanup_function in self.post_parse_functions:
      cleanup_function(namespace)

    return namespace

  @staticmethod
  def _add_checks(default, required):
    if required and default is not None:
      raise ValueError("Required flags do not need defaults.")

  @staticmethod
  def _stringify_choices(choices):
    if choices is None:
      return choices
    output = []
    for i in choices:
      if isinstance(i, (str, int)):
        output.append(str(i))
      else:
        raise ValueError("Could not stringify choices.")
    return output

  @staticmethod
  def _pretty_help(help_text, default, choices, required, var_type):
    prestring = ""
    if choices is not None:
      prestring = "{" + ", ".join([str(var_type(i)) for i in choices]) + "}    "

    if required:
      prestring += "Required."
    elif default is not None:
      prestring += "Default: %(default)s"

    if prestring:
      prestring += "\n"
    return prestring + help_text

  def _shortcut_add(self, var_type, name, short_name=None, nargs=None,
                    default=None, choices=None, required=False, help_text=""):
    """Convenience method to handle common arg addition pattern.

      This method is intended to reduce repeated code for a common arg addition
    pattern.

    Args:
      var_type: The variable type into which the input will be cast.
      name: The full name of the flag. This is also the name that will be used
        for internal representation of the flag through the "dest" argument.
      short_name: Optional abbreviation of the flag.
      nargs:  Number of args that the flag accepts. This is passed directly to
        self.add_argument(), so any legal argparse value (i.e. '?') is allowed.
      default:  Default value if applicable.
      choices:  List of legal choices for the flag.
      required: Boolean of whether this flag must be specified by the caller.
      help_text:  Base text to describe the flag. This text is automatically
        augmented with relevant information such as default and choices.
    """

    self._add_checks(default=default, required=required)

    names = ["--" + name]
    if short_name is not None:
      names = ["-" + short_name] + names

    self.add_argument(
        *names,
        nargs=nargs,
        default=default,
        type=var_type,
        choices=choices,
        required=required,
        help=self._pretty_help(help_text=help_text, default=default,
                               choices=choices, required=required,
                               var_type=var_type),
        metavar=name.upper(),
        dest=name
    )

  def _add_bool(self, name, short_name=None, help_text=""):
    """Convenience function for adding boolean flags.

    Args:
      name: The full name of the flag. This is also the name that will be used
        for internal representation of the flag through the "dest" argument.
      short_name: Optional abbreviation of the flag.
      help_text:  Basic description of the flag.
    """

    names = ["--" + name]
    if short_name is not None:
      names = ["-" + short_name] + names

    self.add_argument(
        *names,
        action="store_true",
        help=help_text,
        dest=name
    )

  #=============================================================================
  # Location Args
  #=============================================================================
  def _add_location_args(self, tmp=False, data=False, separate_train_val=False,
                         model=False):
    """Add arguments for specifying file and folder locations.

    Args:
      tmp:  Allow specification of a temp location.
      data: Allow specification of a data location.
      separate_train_val: Allow specification of separate locations for training
        and validation data.
      model:  Allow specification of a location for model files.
    """
    if tmp:
      self._add_tmp_loc()

    if data:
      (self._add_separate_data_locs() if separate_train_val
       else self._add_unified_data_loc())

    if model:
      self._add_model_loc()

  def _add_tmp_loc(self):
    self._shortcut_add(str, "tmp_loc", "tl", default="/tmp",
                       help_text="A directory to place temporary files.")

  def _add_unified_data_loc(self):
    self._shortcut_add(str, "data_loc", "dl", default="/tmp",
                       help_text="The location where the input data is stored.")

  def _add_separate_data_locs(self):
    self._shortcut_add(str, "train_loc", "tl", default="/tmp",
                       help_text="The location where training data is stored.")
    self._shortcut_add(
        str, "val_loc", "vl", default="/tmp",
        help_text="The location where validation data is stored.")

  def _add_model_loc(self):
    self._shortcut_add(
        str, "model_dir", "md", default="/tmp",
        help_text="The directory where model specific files (event files, "
                  "snapshots, etc.) are stored.")

  #=============================================================================
  # Performance Args
  #=============================================================================
  def _add_performance_args(self, num_parallel_calls=False,
                            channel_format=False):
    """Add arguments for tuning model performance.

    Args:
      num_parallel_calls: Allow the user to specify parallelism for sample
        processing.
      channel_format: Allow the user to specify a the ordering of image
        dimensionality.
    """

    if num_parallel_calls:
      self._add_num_parallel_calls()

    if channel_format:
      self._add_channel_format()

  def _add_num_parallel_calls(self):
    self.post_parse_functions.append(self._parallel_call_check)
    self._shortcut_add(
        int, "num_parallel_calls", "npc", default=-1,
        help_text="The number of records that are processed in parallel "
                  "during input processing. This can be optimized per data "
                  "set but for generally homogeneous data sets, should be "
                  "approximately the number of available CPU cores."
                  "'-1' will use the number of CPU cores present."
    )

  @staticmethod
  def _parallel_call_check(namespace):
    cpu_count = multiprocessing.cpu_count()
    if namespace.num_parallel_calls == -1:
      namespace.num_parallel_calls = cpu_count

  def _add_channel_format(self):
    self.post_parse_functions.append(self._channel_format_check)
    self._shortcut_add(
        str, "channel_format", "cf", default=AUTO_STRING,
        choices=[AUTO_STRING, "channels_first", "channels_last"],
        help_text="A flag to override the data format used in the model. "
                  "channels_first provides a performance boost on GPU but "
                  "is not always compatible with CPU. If left unspecified, "
                  "the data format will be chosen automatically based on "
                  "whether TensorFlow was built for CPU or GPU."
    )

  @staticmethod
  def _channel_format_check(namespace):
    if namespace.channel_format == AUTO_STRING:
      namespace.channel_format = (
          "channels_first" if tf.test.is_built_with_cuda() else "channels_last")

  #=============================================================================
  # Add Common Supervised Learning Args
  #=============================================================================
  def _add_supervised_args(self, train_epochs=False, epochs_per_eval=False,
                           learning_rate=False, batch_size=False):
    """Add arguments for supervised learning.

    Args:
      train_epochs: The number of epochs to train the model.
      epochs_per_eval:  The number of epochs between evaluations of the
        validation data.
      learning_rate: The learning rate for the solver.
      batch_size: The mini-batch size to use during training and evaluation.
    """
    if train_epochs:
      self._add_train_epochs()

    if epochs_per_eval:
      self._add_epochs_per_eval()

    if learning_rate:
      self._add_learning_rate()

    if batch_size:
      self._add_batch_size()

  def _add_train_epochs(self):
    self._shortcut_add(int, "train_epochs", "te", default=1,
                       help_text="The number of epochs to use for training.")

  def _add_epochs_per_eval(self):
    self._shortcut_add(
        int, "epochs_per_eval", "epe", default=1,
        help_text="The number of training epochs to run between evaluations."
    )

  def _add_learning_rate(self):
    self._shortcut_add(
        float, "learning_rate", "lr", default=1.,
        help_text="The learning rate to be used during training."
    )

  def _add_batch_size(self):
    self._shortcut_add(
        int, "batch_size", "bs", default=32,
        help_text="Batch size for training and evaluation."
    )

  #=============================================================================
  # Add Args for Specifying Devices
  #=============================================================================
  def _add_device_args(self, allow_cpu=False, allow_gpu=False, allow_tpu=False,
                       allow_multi_gpu=False):
    """Function for determining which device type args are relevant.

      This function serves two purposes. First, it automates the selection
    of arguments based on which device types are supported. TPUs in particular
    add quite a few arguments. Secondly, it forces all model classes to use the
    same conventions around device specific parameter definition, reducing
    model-to-model variation.

      The exact logic for setting device specific arguments is still being set.

    Args:
      allow_cpu: The model can be set to run on CPU.
      allow_gpu: The model can be set to run on GPU.
      allow_tpu: The model can be set to run on TPU.
      allow_multi_gpu: The model allows multi GPU training as an option.

    Raises:
      ValueError: When an invalid configuration is passed.
    """
    allow_gpu = allow_gpu or allow_multi_gpu  # multi_gpu implies gpu=True

    device_types = []
    if allow_cpu: device_types.append("cpu")
    if allow_gpu: device_types.append("gpu")
    if allow_tpu:
      device_types.append("tpu")
      raise ValueError("tpu args are not ready yet.")

    if not device_types:
      raise ValueError("No legal devices specified.")

    self._add_set_device(device_types=device_types)
    if allow_multi_gpu:
      self._add_multi_gpu()

  def _add_set_device(self, device_types):
    if len(device_types) == 1:
      return  # no need for the user to specify the device

    self.post_parse_functions.append(self._set_device_check)
    self._shortcut_add(
        str, "device", "d", default=AUTO_STRING,
        choices=[AUTO_STRING] + device_types,
        help_text="Primary device for neural network computations. Other tasks "
                  "may occur on other devices. (Generally the CPU.)"
    )

  @staticmethod
  def _set_device_check(namespace):
    if namespace.device == AUTO_STRING:
      namespace.device = "gpu" if tf.test.is_built_with_cuda() else "cpu"

  def _add_multi_gpu(self):
    self._add_bool("multi_gpu", help_text="Run across all available GPUs.")








