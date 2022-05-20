import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from run_classifier_dataset_utils import convert_examples_to_two_features_with_context



def load_train_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    # Prepare data loader
    if task_name == "idiom":
        train_examples = processor.get_train_examples(args.data_dir)

    train_features = convert_examples_to_two_features_with_context(
        train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
    )

    # make features into tensor
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # add additional features for MELBERT_MIP and MELBERT
    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
    all_input_mask_2 = torch.tensor([f.input_mask_2 for f in train_features], dtype=torch.long)
    all_segment_ids_2 = torch.tensor(
        [f.segment_ids_2 for f in train_features], dtype=torch.long
    )
    all_input_ids_3 = torch.tensor([f.input_ids_3 for f in train_features], dtype=torch.long)
    all_input_mask_3 = torch.tensor([f.input_mask_3 for f in train_features], dtype=torch.long)
    all_segment_ids_3 = torch.tensor(
        [f.segment_ids_3 for f in train_features], dtype=torch.long
    )
    all_input_ids_4 = torch.tensor([f.input_ids_4 for f in train_features], dtype=torch.long)
    all_input_mask_4 = torch.tensor([f.input_mask_4 for f in train_features], dtype=torch.long)
    all_segment_ids_4 = torch.tensor(
        [f.segment_ids_4 for f in train_features], dtype=torch.long
    )
    train_data = TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
        all_input_ids_2,
        all_input_mask_2,
        all_segment_ids_2,
        all_input_ids_3,
        all_input_mask_3,
        all_segment_ids_3,
        all_input_ids_4,
        all_input_mask_4,
        all_segment_ids_4,
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    return train_dataloader


def load_dev_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    if task_name == "idiom":
        dev_examples = processor.get_dev_examples(args.data_dir)

    dev_features = convert_examples_to_two_features_with_context(
        dev_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
    )

    logger.info("***** Running evaluation *****")
    
    all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    all_guids = [f.guid for f in dev_features]
    all_idx = torch.tensor([i for i in range(len(dev_features))], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in dev_features], dtype=torch.long)
    all_input_mask_2 = torch.tensor([f.input_mask_2 for f in dev_features], dtype=torch.long)
    all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in dev_features], dtype=torch.long)
    all_input_ids_3 = torch.tensor([f.input_ids_3 for f in dev_features], dtype=torch.long)
    all_input_mask_3 = torch.tensor([f.input_mask_3 for f in dev_features], dtype=torch.long)
    all_segment_ids_3 = torch.tensor([f.segment_ids_3 for f in dev_features], dtype=torch.long)
    all_input_ids_4 = torch.tensor([f.input_ids_4 for f in dev_features], dtype=torch.long)
    all_input_mask_4 = torch.tensor([f.input_mask_4 for f in dev_features], dtype=torch.long)
    all_segment_ids_4 = torch.tensor([f.segment_ids_4 for f in dev_features], dtype=torch.long)
    dev_data = TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
        all_idx,
        all_input_ids_2,
        all_input_mask_2,
        all_segment_ids_2,
        all_input_ids_3,
        all_input_mask_3,
        all_segment_ids_3,
        all_input_ids_4,
        all_input_mask_4,
        all_segment_ids_4,
    )

    # Run prediction for full data
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)

    return all_guids, dev_dataloader


def load_eval_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    if task_name == "idiom":
        eval_examples = processor.get_eval_examples(args.data_dir)

    eval_features = convert_examples_to_two_features_with_context(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
    )

    logger.info("***** Running evaluation *****")
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_guids = [f.guid for f in eval_features]
    all_idx = torch.tensor([i for i in range(len(eval_features))], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in eval_features], dtype=torch.long)
    all_input_mask_2 = torch.tensor([f.input_mask_2 for f in eval_features], dtype=torch.long)
    all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in eval_features], dtype=torch.long)
    all_input_ids_3 = torch.tensor([f.input_ids_3 for f in eval_features], dtype=torch.long)
    all_input_mask_3 = torch.tensor([f.input_mask_3 for f in eval_features], dtype=torch.long)
    all_segment_ids_3 = torch.tensor([f.segment_ids_3 for f in eval_features], dtype=torch.long)
    all_input_ids_4 = torch.tensor([f.input_ids_4 for f in eval_features], dtype=torch.long)
    all_input_mask_4 = torch.tensor([f.input_mask_4 for f in eval_features], dtype=torch.long)
    all_segment_ids_4 = torch.tensor([f.segment_ids_4 for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
        all_idx,
        all_input_ids_2,
        all_input_mask_2,
        all_segment_ids_2,
        all_input_ids_3,
        all_input_mask_3,
        all_segment_ids_3,
        all_input_ids_4,
        all_input_mask_4,
        all_segment_ids_4,
    )
    
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return all_guids, eval_dataloader
