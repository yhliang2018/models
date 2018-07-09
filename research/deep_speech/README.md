# DeepSpeech2 Model
## Overview
This is an implementation of the [DeepSpeech2](https://arxiv.org/pdf/1512.02595.pdf) model. Current implementation is based on the code from the authors' [DeepSpeech code](https://github.com/PaddlePaddle/DeepSpeech) and the implementation in the [MLPerf Repo](https://github.com/mlperf/reference/tree/master/speech_recognition).

## Dataset
The [OpenSLR LibriSpeech Corpus](http://www.openslr.org/12/) are used for model training and evaluation.

Add more details about dataset(TODO).

## Running Code

### Download and preprocess dataset
To download the dataset, issue the following command:
```
python data/download.py
```
Arguments:
  * `--data_dir`: Directory where to download and save the preprocessed data. By default, it is `/tmp/librispeech_data`.

Use the `--help` or `-h` flag to get a full list of possible arguments.

### Train and evaluate model
To train and evaluate the model, issue the following command:
```
python deep_speech.py
```
Arguments:
  * `--model_dir`: Directory to save model training checkpoints. By default, it is `/tmp/ncf/`.
  * `--train_data_dir`: Directory of the training dataset.
  * `--eval_data_dir`: Directory of the evaluation dataset.

There are other arguments about DeepSpeech2 model and training/evaluation process. Use the `--help` or `-h` flag to get a full list of possible arguments with detailed descriptions.
