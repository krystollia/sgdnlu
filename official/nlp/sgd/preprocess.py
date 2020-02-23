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

from official.nlp.sgd import schema
from official.nlp.bert import tokenization

"""
from official.nlp.sgd import preprocess
a = preprocess.XDstc8DataProcessor("./sgddata", "dstc8_tiny").get_dialog_examples("train")
"""

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
    "dstc8_tiny": {
        "train": range(44, 45),
        "dev": range(1, 2),
        "test": range(1, 1)
    },
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


class UnifiedExample(object):
    """A single training/test example for unified sequence classification.
     Includes both span labeler and classifier on the [CLS]
     For examples without an answer, the start and end position are -1.

     It will be used for both intent finding, and slot filling, so that
     we can treat them uniformly.
     when start_position and end position is None, the data is prepared for classifier only.
  """

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 question_text=None,
                 start_position=None,
                 end_position=None,
                 label=None):
        self.is_impossible = None
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.start_position = start_position
        self.end_position = end_position
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.start_position:
            s += ", end_position: %d" % self.end_position
        if self.label:
            s += ", label: %s" % self.label
        return s


def load_dialogues(dialog_json_filepaths):
    """Obtain the list of all dialogues from specified json files."""
    dialogs = []
    for dialog_json_filepath in sorted(dialog_json_filepaths):
        with open(dialog_json_filepath) as f:
            dialogs.extend(json.load(f))
    return dialogs


class Dstc8DataProcessor(object):
    """Data generator for dstc8 dialogues."""

    def __init__(self, dstc8_data_dir, collection):
        self.dstc8_data_dir = dstc8_data_dir
        self._file_ranges = FILE_RANGES[collection]

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

        u = open("utterance.txt", "w")
        dialogs = load_dialogues(dialog_paths)
        schema_path = os.path.join(self.dstc8_data_dir, dataset, "schema.json")
        schemas = schema.Schema(schema_path)

        examples = []
        for dialog_idx, dialog in enumerate(dialogs):
            self._create_examples_from_dialog(dialog, schemas, dataset, i, r, s, u)

    def _create_examples_from_dialog(self, dialog, schemas, dataset, i, r, s, u):
        """Create examples for every turn in the dialog."""
        dialog_id = dialog["dialogue_id"]
        prev_states = {}
        examples = []
        for turn_idx, turn in enumerate(dialog["turns"]):
            # Generate an multiple data points for every frame in every user turn.
            if turn["speaker"] == "USER":
                user_utterance = turn["utterance"]
                u.write("%s\t%s\t%s\n" % (user_utterance, dialog_id, turn_idx))
                user_frames = {f["service"]: f for f in turn["frames"]}
                intent = turn["frames"][0]["state"]["active_intent"]
                service = turn["frames"][0]["service"]
                if intent == "NONE":
                    # If there is no intent "Thanks I am done" we mark there is no description
                    self._write_to_file("Intent", intent, None, user_utterance, i, "1")
                else:
                    # Get the description from the schema and then write it to file
                    intent_desc = schemas.get_service_schema(service).get_intent_description(intent)
                    self._write_to_file("Intent", intent, intent_desc, user_utterance, i, "1")

                # Go through all of the other intents and mark them with INACTIVE tag
                for other_intent in schemas.get_service_schema(service).intents:
                    if other_intent != intent:
                        other_intent_desc = schemas.get_service_schema(service).get_intent_description(other_intent)
                        self._write_to_file("Intent", other_intent, other_intent_desc, user_utterance, i, "0")

                # Go through all requested slots and mark them with yes and write
                for requested_slot in turn["frames"][0]["state"]["requested_slots"]:
                    slot_desc = schemas.get_service_schema(service).get_slot_description(requested_slot)
                    self._write_to_file("Request", requested_slot, slot_desc, user_utterance, r, "1")

                # Go through all unrequested slots and mark them with no and write
                for other_slot in set(schemas.get_service_schema(service).slots) - set(
                        turn["frames"][0]["state"]["requested_slots"]):
                    other_slot_desc = schemas.get_service_schema(service).get_slot_description(other_slot)
                    self._write_to_file("Request", other_slot, other_slot_desc, user_utterance, r, "0")

                # Go through all slot values stated
                for slot in turn["frames"][0]["state"]["slot_values"]:
                    slot_desc = schemas.get_service_schema(service).get_slot_description(slot)
                    # Mark either dontcare or active
                    if turn["frames"][0]["state"]["slot_values"][slot] == "dontcare":
                        self._write_to_file("Slot", slot, slot_desc, user_utterance, s, "1")
                    else:
                        self._write_to_file("Slot", slot, slot_desc, user_utterance, s, "1")

                # All other slots are none
                for other_slot in set(schemas.get_service_schema(service).slots) - set(
                        turn["frames"][0]["state"]["slot_values"]):
                    other_slot_desc = schemas.get_service_schema(service).get_slot_description(other_slot)
                    self._write_to_file("Slot", other_slot, other_slot_desc, user_utterance, s, "0")

        return examples

    def _write_to_file(self, task, name, description, utterance, writer, label):
        if description != None:
            writer.write(
                label + "\t" + "0" + "\t" + "0" + "\t" + task + "-" + name + "-" + description + "\t" + utterance + "\n")
        else:
            writer.write(label + "\t" + "0" + "\t" + "0" + "\t" + task + "-" + name + "\t" + utterance + "\n")


class WDstc8DataProcessor(object):
    """Data generator for dstc8 dialogues."""

    def __init__(self,
                 dstc8_data_dir,
                 collection,
                 vocab_file,
                 do_lower_case,
                 max_seq_length=DEFAULT_MAX_SEQ_LENGTH, ):
        self.dstc8_data_dir = dstc8_data_dir
        self._file_ranges = FILE_RANGES[collection]
        # BERT tokenizer
        self._tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        self._max_seq_length = max_seq_length

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
        dialogs = load_dialogues(dialog_paths)
        schema_path = os.path.join(self.dstc8_data_dir, dataset, "schema.json")
        schemas = schema.Schema(schema_path)

        examples = []
        for dialog_idx, dialog in enumerate(dialogs):
            examples.extend(
                self._create_examples_from_dialog(dialog, schemas, dataset))
        return examples

    def _create_examples_from_dialog(self, dialog, schemas, dataset):
        """Create examples for every turn in the dialog."""
        dialog_id = dialog["dialogue_id"]
        prev_states = {}
        examples = []
        history = []
        turns = dialog["turns"]
        for turn_idx, turn in enumerate(dialog["turns"]):
            # Generate an example for every frame in every user turn.
            if turn["speaker"] == "USER":
                user_utterance = turn["utterance"]
                user_frames = {f["service"]: f for f in turn["frames"]}
                if turn_idx > 0:
                    system_turn = dialog["turns"][turn_idx - 1]
                    system_utterance = system_turn["utterance"]
                    system_frames = {f["service"]: f for f in system_turn["frames"]}
                else:
                    system_utterance = ""
                    system_frames = {}
                turn_id = "{}-{}-{:02d}".format(dataset, dialog_id, turn_idx)
                turn_examples, prev_states, history = self._create_examples_from_turn(
                    turns, turn_id, system_utterance, user_utterance, system_frames,
                    user_frames, prev_states, history, schemas)
                examples.extend(turn_examples)
        return examples

    def _create_examples_from_turn(self, turns, turn_id, system_utterance,
                                   user_utterance, system_frames, user_frames,
                                   prev_states, history, schemas):
        system_tokens, system_alignments, system_inv_alignments = self._tokenize(system_utterance)
        user_tokens, user_alignments, user_inv_alignments = self._tokenize(user_utterance)

        states = {}
        base_example = UnifiedExample(turn_id, user_tokens)
        examples = []
        for service, user_frame in user_frames.items():
            # Create an example for this service.
            example = base_example.copy()
            example.qas_id = "{}-{}".format(turn_id, service)

            system_frame = system_frames.get(service, None)
            state = user_frame["state"]["slot_values"]
            state_update = self._get_state_update(state, prev_states.get(service, {}))
            states[service] = state

            user_span_boundaries = self._find_subword_indices(
                state_update, user_utterance, user_frame["slots"], user_alignments,
                user_tokens, 2 + len(system_tokens))

            if system_frame is not None:
                system_span_boundaries = self._find_subword_indices(
                    state_update, system_utterance, system_frame["slots"],
                    system_alignments, system_tokens, 1)
            else:
                system_span_boundaries = {}

            examples.append(example)
            return examples, states, history

    def _get_state_update(self, current_state, prev_state):
        """This is not nearly enough, we need to move the requested slot and
    among other things """
        state_update = dict(current_state)
        for slot, values in current_state.items():
            if slot in prev_state and prev_state[slot][0] in values:
                # Remove the slot from state if its value didn't change.
                state_update.pop(slot)
        return state_update

    def _find_subword_indices(self, slot_values, utterance, char_slot_spans,
                              alignments, subwords, bias):
        """Find indices for subwords corresponding to slot values."""
        span_boundaries = {}
        for slot, values in slot_values.items():
            # Get all values present in the utterance for the specified slot.
            value_char_spans = {}
            for slot_span in char_slot_spans:
                if slot_span["slot"] == slot:
                    value = utterance[slot_span["start"]:slot_span["exclusive_end"]]
                    start_tok_idx = alignments[slot_span["start"]]
                    end_tok_idx = alignments[slot_span["exclusive_end"] - 1]
                    if 0 <= start_tok_idx < len(subwords):
                        end_tok_idx = min(end_tok_idx, len(subwords) - 1)
                        value_char_spans[value] = (start_tok_idx + bias, end_tok_idx + bias)
            for v in values:
                if v in value_char_spans:
                    span_boundaries[slot] = value_char_spans[v]
                    break
        return span_boundaries

    def _tokenize(self, utterance):
        """
        Tokenize the utterance using word-piece tokenization used by BERT.

        Args:
          utterance: A string containing the utterance to be tokenized.

        Returns:
          bert_tokens: A list of tokens obtained by word-piece tokenization of the
            utterance.
          alignments: A dict mapping indices of characters corresponding to start
            and end positions of words (not subwords) to corresponding indices in
            bert_tokens list.
          inverse_alignments: A list of size equal to bert_tokens. Each element is a
            tuple containing the index of the starting and inclusive ending
            character of the word corresponding to the subword. This list is used
            during inference to map word-piece indices to spans in the original
            utterance.
        """
        utterance = tokenization.convert_to_unicode(utterance)
        # After _naive_tokenize, spaces and punctuation marks are all retained, i.e.
        # direct concatenation of all the tokens in the sequence will be the
        # original string.
        tokens = _naive_tokenize(utterance)
        # Filter out empty tokens and obtain aligned character index for each token.
        alignments = {}
        char_index = 0
        bert_tokens = []
        # These lists store inverse alignments to be used during inference.
        bert_tokens_start_chars = []
        bert_tokens_end_chars = []
        for token in tokens:
            if token.strip():
                subwords = self._tokenizer.tokenize(token)
                # Store the alignment for the index of starting character and the
                # inclusive ending character of the token.
                alignments[char_index] = len(bert_tokens)
                bert_tokens_start_chars.extend([char_index] * len(subwords))
                bert_tokens.extend(subwords)
                # The inclusive ending character index corresponding to the word.
                inclusive_char_end = char_index + len(token) - 1
                alignments[inclusive_char_end] = len(bert_tokens) - 1
                bert_tokens_end_chars.extend([inclusive_char_end] * len(subwords))
            char_index += len(token)
        inverse_alignments = list(
            zip(bert_tokens_start_chars, bert_tokens_end_chars))
        return bert_tokens, alignments, inverse_alignments


def _naive_tokenize(s):
    """Tokenize a string, separating words, spaces and punctuations."""
    # Spaces and punctuation marks are all retained, i.e. direct concatenation
    # of all the tokens in the sequence will be the original string.
    seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", s) if tok]
    return seq_tok


class XDstc8DataProcessor(object):
    """Data generator for dstc8 dialogues."""

    def __init__(self,
                 dstc8_data_dir, collection):
        self.dstc8_data_dir = dstc8_data_dir
        self._file_ranges = FILE_RANGES[collection]

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
        dialogs = load_dialogues(dialog_paths)
        schema_path = os.path.join(self.dstc8_data_dir, dataset, "schema.json")
        schemas = schema.Schema(schema_path)

        examples = []
        for dialog_idx, dialog in enumerate(dialogs):
            examples.extend(
                self._create_examples_from_dialog(dialog, schemas, dataset))
        return examples

    def _create_examples_from_dialog(self, dialog, schemas, dataset):
        """Create examples for every turn in the dialog."""
        dialog_id = dialog["dialogue_id"]
        prev_states = {}
        examples = []
        history = []
        turns = dialog["turns"]
        for turn_idx, turn in enumerate(dialog["turns"]):
            # Generate an example for every frame in every user turn.
            if turn["speaker"] == "USER":
                user_utterance = turn["utterance"]
                user_frames = {f["service"]: f for f in turn["frames"]}
                if turn_idx > 0:
                    system_turn = dialog["turns"][turn_idx - 1]
                    system_utterance = system_turn["utterance"]
                    system_frames = {f["service"]: f for f in system_turn["frames"]}
                else:
                    system_utterance = ""
                    system_frames = {}
                turn_id = "{}-{}-{:02d}".format(dataset, dialog_id, turn_idx)
                turn_examples, prev_states, history = self._create_examples_from_turn(
                    turns, turn_id, system_utterance, user_utterance, system_frames,
                    user_frames, prev_states, history, schemas)
                examples.extend(turn_examples)
        return examples

    def _get_state_update(self, current_state, prev_state):
        """This is not nearly enough, we need to move the requested slot and
        among other things """
        state_update = dict(current_state)
        for slot, values in current_state.items():
            if slot in prev_state and prev_state[slot][0] in values:
                # Remove the slot from state if its value didn't change.
                state_update.pop(slot)
        return state_update

    def _create_examples_from_turn(self, turns, turn_id, system_utterance,
                                   user_utterance, system_frames, user_frames,
                                   prev_states, history, schemas):
        state_update = self._get_state_update()
        if len(user_frames) > 1:
            print(turn_id)
            print(user_utterance)
            print(user_frames)

        return [None], None, None
