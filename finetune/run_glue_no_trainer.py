# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import csv
import jsonlines
import wandb
import numpy as np
import data_formatter

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, ClusterNode, to_tree,  single, cophenet
from scipy.spatial.distance import pdist, squareform

import transformers
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
import torch
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)


logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "axg": ("premise", "hypothesis"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--pred_file", type=str, default=None, help="A csv or a json file containing the test data with the model predictions."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to tokenizer identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=0.1, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the"
                                                         " number of training examples to this value if set."
    )
    parser.add_argument(
        "--max_val_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the"
                                                            " number of validation examples to this value if set."
    )
    parser.add_argument(
        "--max_test_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the"
                                                          " number of test examples to this value if set."
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--do_train", default=False, action='store_true', help="If training the model or not")
    parser.add_argument("--results_file", type=str, required=True, help="file to save the results for the current running")
    parser.add_argument("--neighbors_file", type=str, required=True, help="file to save the neighbors for specific example")
    parser.add_argument("--evaluation_by_group_score_file", type=str, required=True, help="file to save the results for the evaluation_by_pairs")
    parser.add_argument("--prediction_categories", type=str, required=True, help="file to save the category for each model prediction")
    #parser.add_argument("--weighted_average_for_occupations", type=str, required=True, help="file to save the results for the weighted_average_for_occupations")
    # parser.add_argument("--pro_is_correct_sentences_file", type=str, required=True, help="file to save the sentences that the model correct only for the pro stereotype")
    parser.add_argument("--wandb", default=False, action='store_true', help="Save the current run in Wandb")
    parser.add_argument("--wandb_project_name", help="Name of the current run in Wandb")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="hidden dropout prob")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="attention probs dropout prob")
    parser.add_argument("--Axg", default=False, action='store_true', help="Whether the test set is Ax-g or not.")

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def evaluate(model, model_name, data_loader, args, validation=None):
    y_pred = []
    probability = []
    y_true = []
    clustering_labels = []
    metric = load_metric("accuracy")
    counter_all_correct = 0
    counter_all_wrong = 0
    counter_both_correct = 0
    counter_heuristic = 0
    counter_both_correct_skips = 0
    counter_both_not_correct = 0
    counter_both_not_correct_skips = 0
    counter_not_gotcha_correct = 0
    counter_pro_correct_skips = 0
    counter_gotcha_correct = 0
    counter_other = 0
    counter_anti_correct_skips = 0
    counter_neutral_correct = 0
    counter_neutral_correct_skips = 0
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    last_hidden_states_features = np.empty(shape=(data_loader.dataset.num_rows, model.config.hidden_size))
    with torch.no_grad():
        for batch_number, batch in enumerate(data_loader):
            input_ids = batch['input_ids']
            labels = batch['labels']
            attention_mask = batch['attention_mask']
            # if 'roberta' or 'distilbert' not in model_name:
            #     token_type_ids = batch['token_type_ids']
            #     output = model(input_ids, labels=labels, token_type_ids=token_type_ids, attention_mask=attention_mask)
            # else:
            output = model(input_ids, labels=labels, attention_mask=attention_mask)
            predictions = output.logits.argmax(dim=-1)
            last_hidden_states_batch = output.hidden_states[-1]
            cls_embedding = last_hidden_states_batch[:, 0, :].detach()
            last_hidden_states_features[args.per_device_test_batch_size * batch_number: args.per_device_test_batch_size * (batch_number + 1)] = cls_embedding.cpu()
            prob = [max(i) for i in output.logits.softmax(dim=-1).tolist()]
            probability.extend(prob)
            if args.task_name == "mnli":
                # predictions = [1 if p == 1 or p == 2 else 0 for p in predictions]
                predictions = [1 if p == 0 or p == 1 else 0 for p in predictions]
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
            if type(predictions) is not list:
                predictions = predictions.tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())
        # if args.model_name_or_path == 'roberta-large-mnli':
        #     Z = linkage(np.asarray(last_hidden_states_features), method="ward")
        #     clustering_labels.extend(fcluster(Z, t=10, criterion="maxclust"))
    #         for example in [66, 318, 125, 291, 355]:
    #             neighbors = squareform(cophenet(Z))[example].argsort()[:10]
    #             with open(args.neighbors_file, mode='a+', newline='') as neighbors_file:
    #                 neighbors_file = csv.DictWriter(neighbors_file, delimiter=',',fieldnames=['neighbors'])
    #                 neighbors_file.writerow({'neighbors': neighbors})
    # if args.model_name_or_path == 'roberta-large-mnli':
    #     add_cluster_number_to_data_file(clustering_labels, args.pred_file)

    if validation:
        both_not_correct_occupation_dict = {}
        sum_of_both_correct_diff = 0
        sum_of_both_wrong_diff = 0
        sum_of_pro_correct_diff = 0
        sum_of_anti_correct_diff = 0
        sum_of_neutral_correct_diff = 0
        prediction_categories_dict = {'all_correct': 1, 'all_wrong': 2, 'not_gotcha_correct': 3, 'gotcha_correct': 4, 'heuristic': 5, 'other': 6}
        df_prediction = pd.DataFrame(columns=[args.model_name_or_path])

        pair_categories_dict = {'both_correct': 1, 'both_wrong': 2, 'not_gotcha_correct': 3, 'gotcha_correct': 4}
        df_pair_analysis = []

        with jsonlines.open(args.test_file) as reader:
            lines = list(reader)
            gender_mistakes = [False] * len(y_pred)
            quartet_id = 0
            first_pair_id = -1
            second_pair_id = 0
            for i in range(0, len(y_pred), 4):
                quartet_id += 1
                first_pair_id += 2
                second_pair_id += 2
                correct_on_first_sent = y_pred[i] == y_true[i]
                correct_on_second_sent = y_pred[i + 2] == y_true[i + 2]
                correct_on_third_sent = y_pred[i + 1] == y_true[i + 1]
                correct_on_forth_sent = y_pred[i + 3] == y_true[i + 3]

                correct_on_first_pair = correct_on_first_sent and correct_on_second_sent
                correct_on_second_pair = correct_on_third_sent and correct_on_forth_sent
                wrong_on_first_pair = (not correct_on_first_sent) and (not correct_on_second_sent)
                wrong_on_second_pair = (not correct_on_third_sent) and (not correct_on_forth_sent)

                correct_on_both_pairs = correct_on_first_pair and correct_on_second_pair
                wrong_on_both_pairs = wrong_on_first_pair and wrong_on_second_pair

                number_of_correct_sent = correct_on_first_sent + correct_on_second_sent + correct_on_third_sent + correct_on_forth_sent

                if correct_on_both_pairs:
                    counter_all_correct += 1
                    df2 = pd.DataFrame([[prediction_categories_dict['all_correct']] for i in range(4)],
                                                      columns=[args.model_name_or_path])
                    df_prediction = df_prediction.append(df2)

                    df_pair_analysis += [[quartet_id, first_pair_id, 1, 0, 0, 0] if i % 2 == 0 else [quartet_id, second_pair_id, 1,  0, 0, 0] for i in range(4)]

                elif wrong_on_both_pairs:
                    counter_all_wrong += 1
                    df2 = pd.DataFrame([[prediction_categories_dict['all_wrong']] for i in range(4)],
                                       columns=[args.model_name_or_path])
                    df_prediction = df_prediction.append(df2)

                    df_pair_analysis += [[quartet_id, first_pair_id, 0, 1, 0, 0] if i % 2 == 0 else [quartet_id, second_pair_id, 0,  1, 0, 0] for i in range(4)]

                elif (correct_on_first_pair and wrong_on_second_pair) or (correct_on_second_pair and wrong_on_first_pair):
                    counter_heuristic += 1
                    df2 = pd.DataFrame([[prediction_categories_dict['heuristic']] for i in range(4)],
                                       columns=[args.model_name_or_path])
                    df_prediction = df_prediction.append(df2)
                    if correct_on_first_pair and wrong_on_second_pair:
                        df_pair_analysis += [[quartet_id, first_pair_id, 1, 0, 0, 0] if i % 2 == 0 else [quartet_id, second_pair_id, 0, 1, 0, 0] for i in range(4)]
                    else:
                        df_pair_analysis += [
                            [quartet_id, first_pair_id, 0, 1, 0, 0] if i % 2 == 0 else [quartet_id, second_pair_id, 1, 0, 0, 0] for i in range(4)]

                elif number_of_correct_sent == 2:  # bias on both pairs
                    # correct on not_gotcha in the 1st pair
                    if (lines[i]['gotcha'] == "False" and correct_on_first_sent) or \
                            (lines[i + 2]['gotcha'] == "False" and correct_on_second_sent):
                        counter_not_gotcha_correct += 1
                        gotcha = 'not_gotcha_correct'
                        pair_analysis = [quartet_id, first_pair_id, 0, 0, 1, 0]
                        if not correct_on_first_sent:
                            gender_mistakes[i] = True
                        else:
                            gender_mistakes[i + 2] = True
                    else:  # correct on gotcha in the 1st pair
                        counter_gotcha_correct += 1
                        gotcha = 'gotcha_correct'
                        pair_analysis = [quartet_id, first_pair_id, 0, 0, 0, 1]
                        if not correct_on_first_sent:
                            gender_mistakes[i] = True
                        else:
                            gender_mistakes[i + 2] = True
                    first_pair_gotcha = gotcha
                    first_pair_gotcha_list = pair_analysis

                    if (lines[i + 1]['gotcha'] == "False" and correct_on_third_sent) or \
                            (lines[i + 3]['gotcha'] == "False" and correct_on_forth_sent):
                        counter_not_gotcha_correct += 1
                        gotcha = 'not_gotcha_correct'
                        pair_analysis = [quartet_id, second_pair_id, 0, 0, 1, 0]
                        if not correct_on_third_sent:
                            gender_mistakes[i + 1] = True
                        else:
                            gender_mistakes[i + 3] = True
                    else:  # correct on gotcha in the 2nd pair
                        counter_gotcha_correct += 1
                        gotcha = 'gotcha_correct'
                        pair_analysis = [quartet_id, second_pair_id, 0, 0, 0, 1]
                        if not correct_on_third_sent:
                            gender_mistakes[i + 1] = True
                        else:
                            gender_mistakes[i + 3] = True
                    second_pair_gotcha = gotcha
                    second_pair_gotcha_list = pair_analysis
                    gotcha_list = [first_pair_gotcha, second_pair_gotcha, first_pair_gotcha, second_pair_gotcha]
                    pair_gotcha_list = [first_pair_gotcha_list, second_pair_gotcha_list, first_pair_gotcha_list, second_pair_gotcha_list]

                    df2 = pd.DataFrame([[prediction_categories_dict[gotcha]] for gotcha in gotcha_list],
                                       columns=[args.model_name_or_path])
                    df_prediction = df_prediction.append(df2)

                    df_pair_analysis += [gotcha for gotcha in pair_gotcha_list]


                elif number_of_correct_sent == 3 or number_of_correct_sent == 1:
                    if correct_on_first_pair or wrong_on_first_pair:  # and not correct on one of the sentences in the
                        # first pair and wrong on the other
                        if correct_on_first_pair:
                            first_pair_analysis = [quartet_id, first_pair_id, 1, 0, 0, 0]
                        else:
                            first_pair_analysis = [quartet_id, first_pair_id, 0, 1, 0, 0]

                        # on the second pair the model correct on one sentence and wrong on the other one
                        if (lines[i+1]['gotcha'] == "False" and correct_on_third_sent) or \
                                (lines[i + 3]['gotcha'] == "False" and correct_on_forth_sent):
                            counter_not_gotcha_correct += 1
                            counter_other += 1
                            gotcha = 'not_gotcha_correct'
                            second_pair_analysis = [quartet_id, second_pair_id, 0, 0, 1, 0]
                            if not correct_on_third_sent:
                                gender_mistakes[i + 1] = True
                            else:
                                gender_mistakes[i + 3] = True
                        else:
                            counter_gotcha_correct += 1
                            counter_other += 1
                            gotcha = 'gotcha_correct'
                            second_pair_analysis = [quartet_id, second_pair_id, 0, 0, 0, 1]
                            if not correct_on_third_sent:
                                gender_mistakes[i + 1] = True
                            else:
                                gender_mistakes[i + 3] = True

                        categories_list = ['other', gotcha, 'other', gotcha]

                    else:  # correct_on_second_pair or wrong_on_second_pair
                        if correct_on_second_pair:
                            second_pair_analysis = [quartet_id, second_pair_id, 1, 0, 0, 0]
                        else:
                            second_pair_analysis = [quartet_id, second_pair_id, 0, 1, 0, 0]

                        if (lines[i]['gotcha'] == "False" and correct_on_first_sent) or \
                                (lines[i + 2]['gotcha'] == "False" and correct_on_second_sent):
                            counter_not_gotcha_correct += 1
                            counter_other += 1
                            gotcha = 'not_gotcha_correct'
                            first_pair_analysis = [quartet_id, first_pair_id, 0, 0, 1, 0]
                            if not correct_on_first_sent:
                                gender_mistakes[i] = True
                            else:
                                gender_mistakes[i + 2] = True
                        else:
                            counter_gotcha_correct += 1
                            counter_other += 1
                            gotcha = 'gotcha_correct'
                            first_pair_analysis = [quartet_id, first_pair_id, 0, 0, 0, 1]
                            if not correct_on_first_sent:
                                gender_mistakes[i] = True
                            else:
                                gender_mistakes[i + 2] = True
                        categories_list = [gotcha, 'other', gotcha, 'other']

                    pair_gotcha_list = [first_pair_analysis, second_pair_analysis, first_pair_analysis, second_pair_analysis]

                    df2 = pd.DataFrame([prediction_categories_dict[category] for category in categories_list],
                                       columns=[args.model_name_or_path])
                    df_prediction = df_prediction.append(df2)

                    df_pair_analysis += [gotcha for gotcha in pair_gotcha_list]

                else:
                    print("no such stereotype")
                    exit(777)

                # elif y_pred[i] != y_true[i] and y_pred[i + 2] != y_true[i + 2]:
                #     counter_both_not_correct += 1
                # elif y_pred[i + 1] != y_true[i + 1] and y_pred[i + 3] != y_true[i + 3]:
                #     counter_both_not_correct += 1
                #
                # elif (lines[i]['stereotype'] == "pro" and y_pred[i] == y_true[i]) or (
                #         lines[i + 2]['stereotype'] == "pro" and y_pred[i + 2] == y_true[
                #     i + 2]):  # correct on the pro sentence bud not on the second one (anti or neutral)
                #     counter_pro_correct += 1
                # elif (lines[i+1]['stereotype'] == "pro" and y_pred[i+1] == y_true[i+1]) or (
                #         lines[i + 3]['stereotype'] == "pro" and y_pred[i + 3] == y_true[
                #     i + 3]):  # correct on the pro sentence bud not on the second one (anti or neutral)
                #     counter_pro_correct += 1
                # elif (lines[i]['stereotype'] == "anti" and y_pred[i] == y_true[i]) or (
                #         lines[i + 2]['stereotype'] == "anti" and y_pred[i + 2] == y_true[i + 2]):
                #     counter_anti_correct += 1
                # elif (lines[i + 1]['stereotype'] == "anti" and y_pred[i + 1] == y_true[i + 1]) or (
                #         lines[i + 3]['stereotype'] == "anti" and y_pred[i + 3] == y_true[i + 3]):
                #     counter_anti_correct += 1
                # # elif (lines[i]['stereotype'] == "neutral" and y_pred[i] == y_true[i]) or (
                # #         lines[i + 1]['stereotype'] == "neutral" and y_pred[i + 1] == y_true[i + 1]):
                # #     counter_neutral_correct += 1

        add_pair_analysis_and_ids_to_data_file(df_pair_analysis, args.pred_file)
        counters_array = [counter_all_correct, counter_all_wrong,
                          counter_not_gotcha_correct, counter_gotcha_correct, counter_heuristic, counter_other]
        percent_not_gotcha_correct = counter_not_gotcha_correct / (len(y_pred) / 2) * 100
        percent_gotcha_correct = counter_gotcha_correct / (len(y_pred) / 2) * 100
        percent_other = counter_other / (len(y_pred) / 2) * 100
        # percent_neutral_correct = counter_neutral_correct / (len(y_pred) / 2) * 100
        # percent_both_correct = counter_both_correct / (len(y_pred) / 2) * 100
        # percent_both_not_correct = counter_both_not_correct / (len(y_pred) / 2) * 100
        precent_counter_all_correct = counter_all_correct / (len(y_pred) / 4) * 100
        precent_counter_all_wrong = counter_all_wrong / (len(y_pred) / 4) * 100
        precent_heuristic = counter_heuristic / (len(y_pred) / 4) * 100

        # compare_model_predictions_with_and_without_gender(y_pred)
        # add_predictions_and_probability_to_data_file(y_pred, probability, args.pred_file)
        # add_gender_mistake_to_data_file(gender_mistakes, args.pred_file)

        with open(args.evaluation_by_group_score_file, mode='a+', newline='') as evaluation_by_group_score_file:
            fieldnames = ['seed', 'all_correct', 'all_wrong', 'not_gotcha_correct', 'gotcha_correct', 'heuristic', 'other']
            results_writer = csv.DictWriter(evaluation_by_group_score_file, delimiter=',', fieldnames=fieldnames)
            if os.stat(args.evaluation_by_group_score_file).st_size == 0:
                results_writer.writeheader()
            results_writer.writerow({'seed': counters_array})
            results_writer.writerow({'seed': args.seed,
                                     'all_correct': np.round(precent_counter_all_correct, 2),
                                     'all_wrong': np.round(precent_counter_all_wrong, 2),
                                     'not_gotcha_correct': np.round(percent_not_gotcha_correct, 2),
                                     'gotcha_correct': np.round(percent_gotcha_correct, 2),
                                     'heuristic': np.round(precent_heuristic, 2),
                                     'other': np.round(percent_other, 2)})

        # df_prediction.to_csv(f"{args.prediction_categories[:-4]}_tmp_{args.model_name_or_path.replace('/', '_')}.csv", index=False)
        # print(f"{args.prediction_categories[:-4]}_tmp_{args.model_name_or_path.replace('/', '_')}.csv")
        if os.path.exists(args.prediction_categories):
            df = pd.read_csv(args.prediction_categories)
            df = df.reset_index(drop=True)
            df = df.join(df_prediction.reset_index(drop=True))
        else:
            df = df_prediction.reset_index(drop=True)
        df.to_csv(args.prediction_categories, index=False)

        if args.wandb:
            wandb.save(os.path.join(wandb.run.dir, args.evaluation_by_group_score_file))
        # both_not_correct_occupation_dict = {}
        # sum_of_both_correct_diff = 0
        # sum_of_both_wrong_diff = 0
        # sum_of_pro_correct_diff = 0
        # sum_of_anti_correct_diff = 0
        # sum_of_neutral_correct_diff = 0
        #
        # bls_occupation_dict = data_formatter.create_bls_stereotype_dict('dataset/winogender/occupations-stats.tsv',
        #                                                                 'dataset/winobias/female_occupations.txt',
        #                                                                 'dataset/winobias/male_occupations.txt')
        # f = open(args.validation_file)
        # lines = list(csv.reader(f))
        # for i in range(0, len(y_pred), 2):
        #     label = int(lines[i+1][2])
        #     occupation = (lines[i+1][1]).split(sep=" [SEP] ")[label]
        #     if y_pred[i] == y_true[i] and y_pred[i+1] == y_true[i+1]:
        #         counter_both_correct += 1
        #         if occupation in bls_occupation_dict:
        #             sum_of_both_correct_diff += abs(50 - float(bls_occupation_dict[occupation]))
        #         else:
        #             counter_both_correct_skips += 1
        #     elif y_pred[i] != y_true[i] and y_pred[i+1] != y_true[i+1]:
        #         counter_both_not_correct += 1
        #         if occupation in bls_occupation_dict:
        #             sum_of_both_wrong_diff += abs(50 - float(bls_occupation_dict[occupation]))
        #         else:
        #             counter_both_not_correct_skips += 1
        #
        #         if occupation in both_not_correct_occupation_dict:
        #             both_not_correct_occupation_dict[occupation][1] += 1
        #         else:
        #             if occupation in bls_occupation_dict:
        #                 occupation_bls = bls_occupation_dict[occupation]
        #             else:
        #                 occupation_bls = "None"
        #             both_not_correct_occupation_dict[occupation] = [occupation_bls, 1]
        #     elif (lines[i + 1][3] == "pro" and y_pred[i] == y_true[i]) or (lines[i + 2][3] == "pro" and y_pred[i+1] == y_true[i+1]): # correct on the pro sentence bud not on the secoond one (anti or neutral)
        #         counter_pro_correct += 1
        #         if occupation in bls_occupation_dict:
        #             sum_of_pro_correct_diff += abs(50 - float(bls_occupation_dict[occupation]))
        #         else:
        #             counter_pro_correct_skips += 1
        #
        #         # with open(args.pro_is_correct_sentences_file, mode='a+', newline='') as pro_is_correct_sentences:
        #         #     results_writer = csv.writer(pro_is_correct_sentences, delimiter=',')
        #         #     results_writer.writerow(str(args.seed))
        #         #     if lines[i + 1][3] == 'pro':
        #         #         results_writer.writerow(lines[i + 1])
        #         #     else:
        #         #         results_writer.writerow(lines[i + 2])
        #
        #     elif (lines[i + 1][3] == "anti" and y_pred[i] == y_true[i]) or (lines[i + 2][3] == "anti" and y_pred[i+1] == y_true[i+1]):
        #         counter_anti_correct += 1
        #         if occupation in bls_occupation_dict:
        #             sum_of_anti_correct_diff += abs(50 - float(bls_occupation_dict[occupation]))
        #         else:
        #             counter_anti_correct_skips += 1
        #
        #     elif (lines[i + 1][3] == "neutral" and y_pred[i] == y_true[i]) or (lines[i + 2][3] == "neutral" and y_pred[i+1] == y_true[i+1]):
        #         counter_neutral_correct += 1
        #         if occupation in bls_occupation_dict:
        #             sum_of_neutral_correct_diff += abs(50 - float(bls_occupation_dict[occupation]))
        #         else:
        #             counter_neutral_correct_skips += 1
        #     else:
        #         print("no such stereotype")
        #         exit(777)
        # counters_array = [counter_both_correct, counter_both_not_correct, counter_pro_correct, counter_anti_correct, counter_neutral_correct]
        # percent_pro_correct = counter_pro_correct/(len(y_pred)/2) * 100
        # percent_anti_correct = counter_anti_correct/(len(y_pred)/2) * 100
        # percent_neutral_correct = counter_neutral_correct/(len(y_pred)/2) * 100
        # percent_both_correct = counter_both_correct/(len(y_pred)/2) * 100
        # percent_both_not_correct = counter_both_not_correct/(len(y_pred)/2) * 100
        #
        # both_not_correct_occupation_dict = {k: v for k, v in sorted(both_not_correct_occupation_dict.items(),
        #                                                             key=lambda item: item[1][1], reverse=True)}
        # json.dump(both_not_correct_occupation_dict,
        #           open(args.output_dir + '/' + str(args.seed) + '_both_not_correct_occupation_dict.json', 'w'), indent=2)
        # print(both_not_correct_occupation_dict)
        #
        # with open(args.evaluation_by_group_score_file, mode='a+',  newline='') as evaluation_by_group_score_file:
        #     fieldnames = ['seed', 'both_correct', 'both_wrong', 'pro_correct', 'anti_correct', 'neutral_correct']
        #     results_writer = csv.DictWriter(evaluation_by_group_score_file, delimiter=',', fieldnames=fieldnames)
        #     if os.stat(args.evaluation_by_group_score_file).st_size == 0:
        #         results_writer.writeheader()
        #     results_writer.writerow({'seed': counters_array})
        #     results_writer.writerow({'seed': args.seed, 'both_correct': np.round(percent_both_correct, 2),
        #                              'both_wrong': np.round(percent_both_not_correct, 2), 'pro_correct': np.round(percent_pro_correct, 2),
        #                              'anti_correct': np.round(percent_anti_correct, 2), 'neutral_correct': np.round(percent_neutral_correct, 2)})
        #     if args.wandb:
        #         wandb.save(os.path.join(wandb.run.dir, args.evaluation_by_group_score_file))

        # with open(args.weighted_average_for_occupations, mode='a+',  newline='') as weighted_average_for_occupations:
        #     fieldnames = ['seed', 'both_correct', 'both_wrong', 'pro_correct', 'anti_correct', 'neutral_correct']
        #     results_writer = csv.DictWriter(weighted_average_for_occupations, delimiter=',', fieldnames=fieldnames)
        #     if os.stat(args.weighted_average_for_occupations).st_size == 0:
        #         results_writer.writeheader()
        #     if counter_both_not_correct == counter_both_not_correct_skips: # we never calc (50-bls) on both_not_correct
        #         mean_of_both_wrong = None
        #     else:
        #         mean_of_both_wrong = sum_of_both_wrong_diff / (counter_both_not_correct - counter_both_not_correct_skips)
        #     if counter_both_correct == counter_both_correct_skips:
        #         mean_of_both_correct = None
        #     else:
        #         mean_of_both_correct = sum_of_both_correct_diff / (counter_both_correct - counter_both_correct_skips)
        #     if counter_pro_correct == counter_pro_correct_skips:
        #         mean_pro_correct = None
        #     else:
        #         mean_pro_correct = sum_of_pro_correct_diff / (counter_pro_correct - counter_pro_correct_skips)
        #     if counter_anti_correct == counter_anti_correct_skips:
        #         mean_anti_correct = None
        #     else:
        #         mean_anti_correct = sum_of_anti_correct_diff / (counter_anti_correct - counter_anti_correct_skips)
        #     if counter_neutral_correct == counter_neutral_correct_skips:
        #         mean_neutral_correct = None
        #     else:
        #         mean_neutral_correct = sum_of_neutral_correct_diff / (counter_neutral_correct - counter_neutral_correct_skips)
        #
        #     results_writer.writerow({'seed': args.seed, 'both_correct': mean_of_both_correct,
        #                              'both_wrong': mean_of_both_wrong, 'pro_correct': mean_pro_correct,
        #                              'anti_correct': mean_anti_correct, 'neutral_correct': mean_neutral_correct})


    # maps_tags = [0, 0, 1]
    # y_pred = [maps_tags[y] for y in y_pred]
    # gender_parity = count_gender_parity / (len(y_pred) / 2)
    else:
        test_metric = metric.compute()
        logger.info(f"test metric: \n{test_metric}")
        print('Classification Report:')
        print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
        ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
        plt.savefig(args.output_dir + '/Confusion_Matrix.png')
        # plt.clf()
        # plt.close()
        plt.show()
        add_predictions_and_probability_to_data_file(y_pred, probability, args.pred_file)
        if args.wandb:
            wandb.log({"Confusion_Matrix": wandb.Image(ax)})
        return classification_report(y_true, y_pred, labels=[1, 0], digits=4, output_dict=True)['micro avg'][
            'precision']

def main():
    args = parse_args()
    if args.wandb:
        wandb.login(key='df3f15e840793a34a6ce3e92bd8c902dede70f8b')
        wandb.init(project="WinoBias",  tags=["metrics"],  job_type="first",  save_code=True)
        wandb.run.name = args.wandb_project_name

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name,  cache_dir=args.cache_dir if args.cache_dir else None,)
        #raw_datasets = load_dataset('load_dataset.py', args.task_name, cache_dir=args.cache_dir, data_dir="MNLI")
        if args.test_file is not None:
            data_files = {}
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
            test_datasets = load_dataset(extension, data_files=data_files,  cache_dir=args.cache_dir if args.cache_dir else None,)
        pass # TODO continue

    # if args.task_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     raw_datasets = load_dataset(args.task_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name, cache_dir=args.cache_dir if args.cache_dir else None, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir if args.cache_dir else None, use_fast=not args.use_slow_tokenizer)
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        ignore_mismatched_sizes=True
    )

    # Preprocessing the datasets
    if args.Axg:
        sentence1_key_axg, sentence2_key_axg = task_to_keys["axg"]
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_non_stereotype_column_names = [name for name in raw_datasets["train"].column_names if name != "label" and name != "stereotype"]
        if "sentence1" in non_label_non_stereotype_column_names and "sentence2" in non_label_non_stereotype_column_names:
            sentence1_key, sentence2_key, sentence3_key = "sentence1", "sentence2", None
        else:
            if len(non_label_non_stereotype_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_non_stereotype_column_names[0:2] # TODO need to order ot better
            elif len(non_label_non_stereotype_column_names) >= 3: # TODO never gets here
                sentence1_key, sentence2_key, sentence3_key = non_label_non_stereotype_column_names[1:4]
            else:
                sentence1_key, sentence2_key, sentence3_key = non_label_non_stereotype_column_names[0], None, None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}


    if label_to_id is not None: # TODO look
        label_to_id[-1] = -1 # TODO look
        label_to_id["not_entailment"] = 1 # TODO look
        label_to_id["entailment"] = 0 # TODO look
        label_to_id["contradiction"] = 1 # TODO look
        label_to_id["neutral"] = 1 # TODO look
        # TODO check if is the labels are good

    # if label_to_id is not None:
    #     model.config.label2id = label_to_id
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}
    # elif args.task_name is not None and not is_regression:
    #     model.config.label2id = {l: i for i, l in enumerate(label_list)}
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        if sentence1_key and sentence2_key in examples.keys():
            # Tokenize the texts
            texts = ((examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key],)
            )
        elif args.Axg:
            if sentence1_key_axg and sentence2_key_axg in examples.keys():
                # Tokenize the texts
                texts = (
                    (examples[sentence1_key_axg],) if sentence2_key_axg is None else (examples[sentence1_key_axg], examples[sentence2_key_axg],)
                )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                # if args.task_name is not None
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                label2id_axg = {"not_entailment": 1, "neutral": 1, "contradiction": 1, "entailment": 0}
                # In all cases, rename the column to labels because the model will expect that.
                examples["label"] = [label2id_axg[examples["label"][i]] if examples["label"][i] in label2id_axg else examples["label"][i] for i in range(len(examples["label"]))]
                result["labels"] = examples["label"]

        return result

    train_column_names = raw_datasets["train"].column_names
    if args.test_file is not None:
        test_column_names = test_datasets["test"].column_names
    else:
        test_column_names = train_column_names

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=train_column_names
    )
    if args.test_file is not None:
        processed_test_datasets = test_datasets.map(
            preprocess_function, batched=True, remove_columns=test_column_names
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    if args.test_file is not None:
        test_dataset = processed_test_datasets["test"]
    else:
        test_dataset = processed_datasets["validation"]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(args.max_val_samples))
    if args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(args.max_test_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if args.num_warmup_steps is not None and args.num_warmup_steps <= 1: # else the num_warmup_steps remains as is
            args.num_warmup_steps = int(args.num_warmup_steps * args.max_train_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Train!
    if args.do_train is True:
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        train_loss_list = []
        val_acc_list = []
        train_acc_list = []
        best_validation_accuracy = 0.
        for epoch in range(args.num_train_epochs):
            model.train()
            accumulation_loss = 0.0
            last_batch_accumulation_step = len(train_dataloader) % args.gradient_accumulation_steps
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss # TODO check if its ok
                if step+1 <= len(train_dataloader) - last_batch_accumulation_step: # we will complete a whole accumulation
                    accumulation_normalization = args.gradient_accumulation_steps
                else: # we divide by the accumulation steps left
                    accumulation_normalization = last_batch_accumulation_step
                loss = loss / accumulation_normalization
                accumulation_loss += float(loss)
                accelerator.backward(loss)
                if (step+1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    if args.wandb:
                        wandb.log({'loss': float(accumulation_loss)})
                    train_loss_list.append(float(accumulation_loss))
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    accumulation_loss = 0.0
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

                if completed_steps >= args.max_train_steps:
                    break
            train_metric = metric.compute()
            train_acc_list.append(train_metric["accuracy"])
            logger.info(f"epoch {epoch}: train accuracy {train_metric}")
            if args.wandb:
                wandb.log({'accuracy': train_metric["accuracy"]})

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

            eval_metric = metric.compute()
            cur_validation_accuracy = eval_metric["accuracy"]
            val_acc_list.append(cur_validation_accuracy)
            logger.info(f"epoch {epoch}: validation accuracy {cur_validation_accuracy}")

            if eval_metric["accuracy"] > best_validation_accuracy:
                best_validation_accuracy = cur_validation_accuracy
                if args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

        model = model.from_pretrained(args.output_dir)
    #
    # if args.task_name == "mnli":
    #     # Final evaluation on mismatched validation set
    #     eval_dataset = processed_datasets["validation_mismatched"]
    #     eval_dataloader = DataLoader(
    #         eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    #     )
    #     eval_dataloader = accelerator.prepare(eval_dataloader)
    #
    #     model.eval()
    #     for step, batch in enumerate(eval_dataloader):
    #         outputs = model(**batch)
    #         predictions = outputs.logits.argmax(dim=-1)
    #         metric.add_batch(
    #             predictions=accelerator.gather(predictions),
    #             references=accelerator.gather(batch["labels"]),
    #         )
    #
    #     eval_metric = metric.compute()
    #     logger.info(f"mnli-mm: {eval_metric}")

    # model = model.from_pretrained("roberta_base/")
    accuracy = evaluate(model, args.model_name_or_path, test_dataloader, args)
    print(accuracy)
    if float(accuracy) >= 0.4:
        evaluate(model, args.model_name_or_path, test_dataloader, args, True)
        if args.do_train:
            with open(args.results_file, mode='a+', newline='') as results_file:
                fieldnames = ['seed', 'train_loss', 'train_acc', 'val_acc', 'accuracy']
                results_writer = csv.DictWriter(results_file, delimiter=',', fieldnames=fieldnames)
                if os.stat(args.results_file).st_size == 0:
                    results_writer.writeheader()
                results_writer.writerow({'seed': args.seed, 'train_loss': train_loss_list, 'train_acc': train_acc_list,
                                         'val_acc': val_acc_list, 'accuracy': accuracy})
        else:
            with open(args.results_file, mode='a+', newline='') as results_file:
                fieldnames = ['accuracy']
                results_writer = csv.DictWriter(results_file, delimiter=',', fieldnames=fieldnames)
                if os.stat(args.results_file).st_size == 0:
                    results_writer.writeheader()
                results_writer.writerow({'accuracy': accuracy})


def add_predictions_and_probability_to_data_file(y_pred, probabilities, data_file):
    lines = []
    args = parse_args()
    with jsonlines.open(data_file) as f:

        for i, line in enumerate(f.iter()):
            line['prediction_' + args.model_name_or_path] = y_pred[i]
            line['probability_' + args.model_name_or_path] = probabilities[i]
            lines.append(line)

    with jsonlines.open(data_file, mode='w') as writer:
        for line in lines:
                writer.write(line)


def add_pair_analysis_and_ids_to_data_file(details_list, data_file):
    lines = []
    pair_categories_dict = {'quartet_id': 0, 'pair_id': 1, 'both_correct': 2, 'both_wrong': 3, 'not_gotcha_correct': 4, 'gotcha_correct': 5}
    with jsonlines.open(data_file) as f:

        for i, line in enumerate(f.iter()):
            print(details_list[i])
            line['quartet_id'] = details_list[i][pair_categories_dict['quartet_id']]
            line['pair_id'] = details_list[i][pair_categories_dict['pair_id']]
            line['both_correct'] = details_list[i][pair_categories_dict['both_correct']]
            line['both_wrong'] = details_list[i][pair_categories_dict['both_wrong']]
            line['not_gotcha_correct'] = details_list[i][pair_categories_dict['not_gotcha_correct']]
            line['gotcha_correct'] = details_list[i][pair_categories_dict['gotcha_correct']]
            lines.append(line)

    with jsonlines.open(data_file, mode='w') as writer:
        for line in lines:
                writer.write(line)



def add_cluster_number_to_data_file(clustering_labels, data_file):
    lines = []
    args = parse_args()
    with jsonlines.open(data_file) as f:

        for i, line in enumerate(f.iter()):
            line['cluster_' + args.model_name_or_path] = int(clustering_labels[i])
            lines.append(line)

    with jsonlines.open(data_file, mode='w') as writer:
        for line in lines:
                writer.write(line)


def add_gender_mistake_to_data_file(gender_mistake, data_file):
    lines = []
    args = parse_args()
    with jsonlines.open(data_file) as f:

        for i, line in enumerate(f.iter()):
            line['gender_mistake_' + args.model_name_or_path] = bool(gender_mistake[i])
            lines.append(line)

    with jsonlines.open(data_file, mode='w') as writer:
        for line in lines:
                writer.write(line)


def compare_model_predictions_with_and_without_gender(y_pred):
    counter_same_prediction = 0
    for i in range(0, len(y_pred), 2):
        if y_pred[i] == y_pred[i+1]:
            counter_same_prediction += 1

    print(f"the percent of the same model prediction for the sentences with and without gender is: {counter_same_prediction/(len(y_pred)/2)}")





if __name__ == "__main__":
    main()
