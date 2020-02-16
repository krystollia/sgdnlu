# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Dataset reader and tokenization-related utilities for baseline model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import json
import os
import re

import schema



# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
EMBEDDING_DIMENSION = 768
# Maximum allowed number of categorical trackable slots for a service.
MAX_NUM_CAT_SLOT = 6
# Maximum allowed number of non-categorical trackable slots for a service.
MAX_NUM_NONCAT_SLOT = 12
# Maximum allowed number of values per categorical trackable slot.
MAX_NUM_VALUE_PER_CAT_SLOT = 11
# Maximum allowed number of intents for a service.
MAX_NUM_INTENT = 4
STR_DONTCARE = "dontcare"
# The maximum total input sequence length after WordPiece tokenization.
DEFAULT_MAX_SEQ_LENGTH = 128

# These are used to represent the status of slots (off, active, dontcare) and
# intents (off, active) in dialogue state tracking.
STATUS_OFF = 0
STATUS_ACTIVE = 1
STATUS_DONTCARE = 2

FILE_RANGES = {
    "dstc8_single_domain": {
        "train": range(1, 44),
        "dev": range(1, 8),
        "test": range(1, 12)
    },
    "dstc8_multi_domain": {
        "train": range(44, 128),
        "dev": range(8, 21),
        "test": range(12, 35)
    },
    "dstc8_all": {
        "train": range(1, 128),
        "dev": range(1, 21),
        "test": range(1, 35)
    }
}

# Name of the file containing all predictions and their corresponding frame
# metrics.
PER_FRAME_OUTPUT_FILENAME = "dialogues_and_metrics.json"


def load_dialogues(dialog_json_filepaths):
  """Obtain the list of all dialogues from specified json files."""
  dialogs = []
  for dialog_json_filepath in sorted(dialog_json_filepaths):
    with open(dialog_json_filepath) as f:
      dialogs.extend(json.load(f))
  return dialogs


class Dstc8DataProcessor(object):
  """Data generator for dstc8 dialogues."""

  def __init__(self,
               dstc8_data_dir,
               train_file_range,
               dev_file_range,
               test_file_range):
    self.dstc8_data_dir = dstc8_data_dir
    self._file_ranges = {
        "train": train_file_range,
        "dev": dev_file_range,
        "test": test_file_range,
    }

  def get_dialog_examples(self, dataset):
    """Return a list of `InputExample`s of the data splits' dialogues.
    Args:
      dataset: str. can be "train", "dev", or "test".
    Returns:
      examples: a list of `InputExample`s.
    """
    dialog_paths = [
        os.path.join(self.dstc8_data_dir, dataset,
                     "dialogues_{:03d}.json".format(i))
        for i in self._file_ranges[dataset]
    ]
    i = open("intent.txt", "w")
    r = open("response.txt", "w")
    s = open("status.txt", "w")

    dialogs = load_dialogues(dialog_paths)
    schema_path = os.path.join(self.dstc8_data_dir, dataset, "schema.json")
    schemas = schema.Schema(schema_path)

    examples = []
    for dialog_idx, dialog in enumerate(dialogs):
      self._create_examples_from_dialog(dialog, schemas, dataset, i, r, s)

  def _create_examples_from_dialog(self, dialog, schemas, dataset, i, r, s):
    """Create examples for every turn in the dialog."""
    dialog_id = dialog["dialogue_id"]
    prev_states = {}
    examples = []
    for turn_idx, turn in enumerate(dialog["turns"]):
      # Generate an example for every frame in every user turn.
      if turn["speaker"] == "USER":
        user_utterance = turn["utterance"]
        user_frames = {f["service"]: f for f in turn["frames"]}
        intent = turn["frames"][0]["state"]["active_intent"]
        service = turn["frames"][0]["service"]
        intent_desc = self._get_description(schemas.get_service_schema(service).intents, intent)["description"]
        self._write_to_file("Intent", intent, intent_desc, user_utterance, i)
        for requested_slot in turn["frames"][0]["state"]["requested_slots"]:
          slot_desc = self._get_description(schemas.get_service_schema(service).slots, requested_slot)["description"]
          self._write_to_file("Request", requested_slot, slot_desc, user_utterance, r)

        for slot in turn["frames"][0]["state"]["slot_values"]:
          slot_desc = self._get_description(schemas.get_service_schema(service).slots, slot)["description"]
          self._write_to_file("Slot", slot, slot_desc, user_utterance, s)

        examples.extend(turn_examples)
    return examples

  def _get_description(self, listofidct, name):
    print(listofidct)
    for element in listofidct:
      print(element)
      if element["name"] == name:
        return element
    return None

  def _write_to_file(self, name, label, description, utterance, writer):
    if label != None:
      writer.write("[CLS] "+name+"-"+label+"-"+description+" [SEP] "+ utterance)
    else:
      writer.write("[CLS] "+name+"-"+description+" [SEP] "+ utterance)
  

a = Dstc8DataProcessor("/Users/stephenwu/nlu/sgddata", 
        FILE_RANGES["dstc8_all"]["train"],
        FILE_RANGES["dstc8_all"]["dev"],
        FILE_RANGES["dstc8_all"]["test"])
a.get_dialog_examples("train")
