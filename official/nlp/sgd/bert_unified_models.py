# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""BERT models that are compatible with TF 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.nlp.sgd import bert_unified_labeler
from official.nlp.bert import bert_models

def unified_model(bert_config,
                  num_labels,
                  max_seq_length,
                  initializer=None):
  """BERT unified model in functional API style.

  Construct a Keras model for predicting `num_labels` outputs from an input with
  maximum sequence length `max_seq_length`, as well as

  Args:
    bert_config: BertConfig or AlbertConfig, the config defines the core
      BERT or ALBERT model.
    num_labels: integer, the number of classes.
    max_seq_length: integer, the maximum input sequence length.
    final_layer_initializer: Initializer for final dense layer. Defaulted
      TruncatedNormal initializer.
    hub_module_url: TF-Hub path/url to Bert module.

  Returns:
    Combined prediction model (words, mask, type) -> (one-hot labels)
    BERT sub-model (words, mask, type) -> (bert_outputs)
  """
  if initializer is None:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)

  bert_encoder = bert_models.get_transformer_encoder(bert_config, max_seq_length)
  return bert_unified_labeler.BertUnifiedLabeler(
      bert_encoder,
      num_classes=num_labels,
      initializer=initializer,
      dropout_rate=bert_config.hidden_dropout_prob), bert_encoder

