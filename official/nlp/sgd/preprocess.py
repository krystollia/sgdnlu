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
import copy
import json
import os
import re

from official.nlp.sgd import schema
from official.nlp.bert import tokenization

# this script is based on the sgd baseline implementation.

"""
We need to handle these different cases.
"CONFIRM",
"GOODBYE",
"INFORM",
"INFORM_COUNT",
"NOTIFY_FAILURE",
"NOTIFY_SUCCESS",
"OFFER",
"OFFER_INTENT",
"REQUEST",
"REQ_MORE",
"""

#
# The current sgd is designed with assumption that user/system utterances are first order
# operation on the frames in term of fillings. This view, while works, ignored many angles
# that apparently different use cases are actually the same things, which can greatly
# reduce the modeling complexity,
#
# The main issue is we need to treat the frame filling as a high order process, meaning we
# stack more than one layer of filling operators, and these operators can operate the frame
# filling in the composite fashion.
#
# 1. requested-slot should be small intent for query some slots, which has nothing to do with
#    host frame. The difference there is the slot name for the origin intent because slot value
#    value for this meta intent.
# 2. there is single value list selection skills that is hidden in many conversations. This is
#    started by the user requesting some slots, then system start to offer the item off the list.
#    User can now refine it, or asking for details, pick one or reject the current item (which
#    in turn look for another one). Again this is an high order/meta constructs which need to
#    be handled in a decomposable fashion in order to greatly increase the reusability.
#


"""
To generate the training examples, we have to go over all the possible the scenarios. 
The main the assumption of the logical conversational interface for services is that we build
common understanding of what user wants in form of frame filling.

Frame are stacked together to build the common understanding, each is serving as the context
for the frame on the top.

"""


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

    def __init__(self, qas_id, doc_tokens):
        self.is_intent = True
        self.service = None
        self.slot = None
        self.context = None
        self.qas_id = qas_id
        self.question_text = None
        self.doc_tokens = doc_tokens
        self.start_position = None
        self.end_position = None
        self.label = None
        self.weight = 1.0

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
        if self.service:
            s += ", service: %s" % self.service
        if self.slot:
            s += ", slot: %s" % self.slot
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


class XDstc8DataProcessor(object):
    """Data generator for dstc8 dialogues."""

    def __init__(self,
                 dstc8_data_dir,
                 collection,
                 vocab_file="./models/uncased_L-12_H-768_A-12/vocab.txt",
                 do_lower_case=True,
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
        sexamples = []
        counts = collections.defaultdict(int)
        for dialog_idx, dialog in enumerate(dialogs):
            e, s = self._create_examples_from_dialog(dialog, schemas, counts, dataset)
            examples.extend(e)
            sexamples.extend(s)
        print(counts)
        return examples, sexamples

    def _create_examples_from_dialog(self, dialog, schemas, counts, dataset):
        """Create examples for every turn in the dialog."""
        dialog_id = dialog["dialogue_id"]
        prev_states = {}
        iexamples = []
        sexamples = []
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
                turn_i_examples, turn_s_examples, prev_states, history = self._create_examples_from_turn(
                    turns, turn_id, system_utterance, user_utterance, system_frames,
                    user_frames, prev_states, history, schemas, counts)
                iexamples.extend(turn_i_examples)
                sexamples.extend(turn_s_examples)
        return iexamples, sexamples

    def _count_contained(self, update_keys, slots):
        """
        Test whether the update is fully or partially contained in the slots.
        :param updates:
        :param slots:
        :return: number of slot are contained in the update.
        """
        count = 0
        for match in slots:
            if match["slot"] in update_keys: count += 1
        return count

    def _filter(self, actions, action_type):
        res = []
        for action in actions:
            if action["act"] == action_type:
                res.append(action)
        return res

    def _get_action_types(self, actions):
        """ Return all the action types from system actions"""
        res = set()
        for action in actions:
            res.add(action["act"])
        return res

    def _build_negative_intent_examples(self, iexamples, list_of_intents):
        res = []
        for intent in list_of_intents:
            res.add()
        iexamples.extend([])


    def _create_examples_from_turn(self, turns, turn_id, system_utterance,
                                   user_utterance, system_frames, user_frames,
                                   prev_states, history, schemas, counts):
        system_tokens, system_alignments, system_inv_alignments = self._tokenize(system_utterance)
        user_tokens, user_alignments, user_inv_alignments = self._tokenize(user_utterance)

        states = {}
        base_example = UnifiedExample(turn_id, user_tokens)
        iexamples = []
        sexamples = []

        multiframe = len(user_frames) != 1
        for service, user_frame in user_frames.items():
            counts["total"] += 1

            # Create an example for this service.
            example = copy.copy(base_example)
            example.qas_id = "{}-{}".format(turn_id, service)

            service_schema = schemas.get_service_schema(service)

            system_frame = system_frames.get(service, None)
            curr_state = user_frame["state"]
            prev_state = prev_states.get(service, {})
            state_update = self._get_state_update(curr_state, prev_state)
            states[service] = curr_state

            # if the state update in the slots, we have training example for this intent
            user_slots = [] if "slots" not in user_frame.keys() else user_frame["slots"]

            user_span_boundaries = self._find_subword_indices(
                state_update, user_utterance, user_frame["slots"], user_alignments,
                user_tokens, 2 + len(system_tokens))

            if system_frame is not None:
                system_span_boundaries = self._find_subword_indices(
                    state_update, system_utterance, system_frame["slots"],
                    system_alignments, system_tokens, 1)
            else:
                system_span_boundaries = {}

            # we need to first figure out when is the uptick of the intent.
            old_active_intent = "" if len(prev_state) == 0 else prev_state["active_intent"]
            new_active_intent = curr_state["active_intent"]


            # for intent, we check couple things:
            # 1. if there is the no active state for this service last user turn. And we saw all
            #    all the slots are mentioned here, it is an direct hit, we should create the
            #    training example as is.
            # whether there is offer: offer has an yes or no expectation.
            mentioned_count = self._count_contained(state_update.keys(), user_slots)
            system_actions = [] if system_frame is None or "actions" not in system_frame.keys() else system_frame["actions"]
            offerred = self._filter(system_actions, "OFFER")
            requested = self._filter(system_actions, "REQUEST")
            confirmed = self._filter(system_actions, "CONFIRM")
            success_notified = self._filter(system_actions, "NOTIFY_SUCCESS")
            more_requested = self._filter(system_actions, "REQ_MORE")
            intent_offered = self._filter(system_actions, "OFFER_INTENT")
            informed = self._filter(system_actions, "INFORM")

            # The first import things is to partition the dialog space,
            # based on state change and system actions.
            action_types = self._get_action_types(system_actions)
            if old_active_intent != new_active_intent:
                counts["active_intent_change"] += 1
                # based on the initial study, although there are four different cases
                # here, but only three exist: and two of them are equivalent. Among them
                # #2 is the same as #1, as if system turn has nothing to do with a service,
                # then it is considered to be terminated, so that we need to treat the service
                # as new mention as well.
                #
                # 1. len(action_types) == 0 and old_active_intent == "":
                # 2. len(action_types) == 0 and old_active_intent != "": the same as 1.
                # 3. len(action_types) != 0 and old_active_intent == "": does not exist
                # 4. len(action_types) != 0 and old_active_intent != "":
                #

                if len(action_types) == 0:
                    if old_active_intent == "":
                        counts["active_intent_change_00"] += 1
                    else:
                        counts["active_intent_change_01"] += 1
                    # This is the always consider to be new start.
                    
                else:
                    if old_active_intent == "":
                        raise ValueError("This combination is not right.")
                    else:
                        counts["active_intent_change_11"] += 1

                    # this is where we handle the different system action types.

            else:
                counts["active_intent_not_change"] += 1
                # we handle different system action types here.



            continue
            # to make thing easy to check, we only focus on 00014.
            if turn_id.find("00014") < 0: continue

            # There are two major scenarios that we care about, we detected an active
            # intent change, or we did not.

            #if old_active_intent != new_active_intent:

            # there are two kind of start of intents: user mention, or intent offering.



            # the state change is fully explained by user utterance.
            if len(offerred) == 0 and len(requested) == 0 and len(confirmed) == 0:
                if mentioned_count > 0 and len(state_update) != 0:
                    example.question_text = service_schema.description
                    example.label = 1
                    # We use weight to nudge the decision a bit as we do not
                    # have ground truth one way or another.
                    if old_active_intent != new_active_intent:
                        example.weight = 1.0
                    else:
                        example.weight = 0.5
                    continue
                elif len(success_notified) != 0:
                    # we do not need to do much? or we check the ack as well.
                    example.question_text = "acknowledge"
                    example.context = "Notified success"
                    example.label = 1
                    continue
                elif len(more_requested) != 0:
                    example.question_text = "no"
                    example.context = "request more"
                    example.label = 1
                    continue
                elif len(intent_offered) != 0:
                    intent = intent_offered[0]["values"]
                    example.context = "binary offer"
                    example.label = 1
                    if intent == new_active_intent:
                        example.question_text = "yes"
                    else:
                        example.question_text = "no"
                    continue
                elif len(informed) != 0:
                    example.question_text = "acknowledge"
                    example.context = "Notified success"
                    example.label = 1
                    continue
                else:
                    print(system_actions)
                    print(mentioned_count)
                    print(turn_id)
                    print(service)
                    print(len(user_frames))
                    print(prev_state)
                    print(curr_state)
                    print(state_update)
                    print(user_utterance)
                    print("\n")
                    raise ValueError("something is wrong here?")

            # Now deal with offer, if the system have offered, and user accepted or not.
            if len(offerred) != 0 and len(requested) == 0 and len(confirmed) == 0:
                accepted_offer_count = self._count_contained(state_update, system_actions)
                example.context = "listed offers"
                example.question_text = "take the offer"
                # Now use accepted some offer, so it is a yes.
                if accepted_offer_count != 0:
                    example.label = 1
                    continue
                else:
                    example.label = 0
                    continue

            if len(offerred) == 0 and len(requested) != 0 and len(confirmed) == 0:
                example.context = requested
                continue

            if len(offerred) == 0 and len(requested) == 0 and len(confirmed) != 0:
                example.context = requested
                continue
            else:
                print(confirmed)
                print(mentioned_count)
                print(len(state_update))
                print(turn_id)
                print(service)
                print(len(user_frames))
                print(state_update)
                print(user_utterance)
                print("\n")
                print("\n")
                print("\n")
                raise ValueError("something is wrong here?")





            print(mentioned_count)
            print(len(state_update))
            print(turn_id)
            print(service)
            print(len(user_frames))
            print(state_update)
            print(user_utterance)
            print("\n")
            print("\n")
            print("\n")

            iexamples.append(example)
        return iexamples, sexamples, states, history

    def _get_state_update(self, current_state, prev_state):
        """
        This is not nearly enough, we need to move the requested slot and
        among other things
        """
        state_update = dict(current_state["slot_values"])
        prev_state_svs = {} if "slot_values" not in prev_state else prev_state["slot_values"]
        for slot, values in current_state["slot_values"].items():
            if slot in prev_state_svs and prev_state_svs[slot][0] in values:
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


if __name__ == "__main__":
    _, _ = XDstc8DataProcessor("./sgddata", "dstc8_all").get_dialog_examples("train")
