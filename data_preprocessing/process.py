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
      # Generate an multiple data points for every frame in every user turn.
      if turn["speaker"] == "USER":
        user_utterance = turn["utterance"]
        user_frames = {f["service"]: f for f in turn["frames"]}
        intent = turn["frames"][0]["state"]["active_intent"]
        service = turn["frames"][0]["service"]
        if intent == "NONE":
          #If there is no intent "Thanks I am done" we mark there is no description
          self._write_to_file("Intent", intent, None, user_utterance, i, "ACTIVE")
        else:
          #Get the description from the schema and then write it to file
          intent_desc = schemas.get_service_schema(service).get_intent_description(intent)
          self._write_to_file("Intent", intent, intent_desc, user_utterance, i, "ACTIVE")

        #Go through all of the other intents and mark them with INACTIVE tag
        for other_intent in schemas.get_service_schema(service).intents:         
          if other_intent != intent:
            other_intent_desc = schemas.get_service_schema(service).get_intent_description(other_intent)
            self._write_to_file("Intent", other_intent, other_intent_desc, user_utterance, i, "INACTIVE")
        
        #Go through all requested slots and mark them with yes and write
        for requested_slot in turn["frames"][0]["state"]["requested_slots"]:
          slot_desc = schemas.get_service_schema(service).get_slot_description(requested_slot)
          self._write_to_file("Request", requested_slot, slot_desc, user_utterance, r, "YES")

        #Go through all unrequested slots and mark them with no and write
        for other_slot in set(schemas.get_service_schema(service).slots)-set(turn["frames"][0]["state"]["requested_slots"]):
          other_slot_desc = schemas.get_service_schema(service).get_slot_description(other_slot)
          self._write_to_file("Request", other_slot, other_slot_desc, user_utterance, r, "NO")

        #Go through all slot values stated
        for slot in turn["frames"][0]["state"]["slot_values"]:
          slot_desc = schemas.get_service_schema(service).get_slot_description(slot)
          #Mark either dontcare or active
          if turn["frames"][0]["state"]["slot_values"][slot] == "dontcare":
            self._write_to_file("Slot", slot, slot_desc, user_utterance, s, "DONTCARE")
          else:
            self._write_to_file("Slot", slot, slot_desc, user_utterance, s, "ACTIVE")

        #All other slots are none
        for other_slot in set(schemas.get_service_schema(service).slots)-set(turn["frames"][0]["state"]["slot_values"]):
          other_slot_desc = schemas.get_service_schema(service).get_slot_description(other_slot)
          self._write_to_file("Slot", other_slot, other_slot_desc, user_utterance, s, "NONE")
    
    return examples


  def _write_to_file(self, task, name, description, utterance, writer, label):
    if description != None:
      writer.write(task+"-"+name+"-"+description+"\t"+ utterance+"\t"+label+"\n")
    else:
      writer.write(task+"-"+name+"\t"+ utterance+ "\t"+label+"\n")
  

a = Dstc8DataProcessor("/Users/stephenwu/nlu/sgddata", 
        FILE_RANGES["dstc8_all"]["train"],
        FILE_RANGES["dstc8_all"]["dev"],
        FILE_RANGES["dstc8_all"]["test"])
a.get_dialog_examples("train")
