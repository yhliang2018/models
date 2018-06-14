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
"""Defines Deep Speech model parameters."""


MODEL_PARAMS =dict(
    # Audio sampling parameters
    sample_rate=16000,  # sample rate
    window_size=0.02,  # window size for spectrogram in seconds
    window_stride=0.01,  # window stride for spectrogram in seconds
    window="hamming",  # window type to generate spectrogram

    # Audio noise parameters
    noise_dir=None,  # directory to inject noise
    noise_prob=0.4,  # probability of noise being added per sample
    noise_min=0.0,  # minimum noise level to sample from (1.0 means all noise and no original signal)
    noise_max=0.5,  # maximum noise level to sample from (1.0 means all noise and no original signal)

    # Dataset location
    labels_path="../labels.json",  # Contains all characters for prediction
    train_manifest="../libri_train_manifest.csv",  # relative path to train manifest is download_dataset is used
    val_manifest="../libri_val_manifest.csv",  # relative path to val manifest is download_dataset is used

    # Model parameters
    hidden_size=800,  # Hidden size of RNNs
    hidden_layers=5,  # Number of RNN layers
    bias=True,  # Use biases
    rnn_type="gru",  #Type of the RNN. rnn|gru|lstm are supported
    rnn_act_type="tanh",  #Type of the activation within RNN. tanh | relu are supported

    # Training parameters
    epochs=10,  # Number of training epochs
    learning_anneal=1.1,  # Annealing applied to learning rate every epoch
    lr=0.0003,  # initial learning rate
    momentum=0.9,  # momentum
    max_norm=200,  # Norm cutoff to prevent explosion of gradients
    l2=0,  # L2 regularization
    batch_size=5,  # Batch size for training
    augment=True,  # Use random tempo and gain perturbations
    exit_at_acc=True,  # Exit at given target accuracy
)
