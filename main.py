import os
import sys
import pickle
import random
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm, trange
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from utils import Config, Logger, make_log_dir
from modeling import AutoModelForSequenceClassification_SPV_MIP

from run_classifier_dataset_utils import processors, output_modes, compute_metrics
from data_loader import load_train_data, load_eval_data, load_dev_data
from data_preproc import *

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"


def main():
    # read configs
    config = Config(main_conf_path="./")


    argv = sys.argv[1:]
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = " ".join(sys.argv[1:]).split(" ")
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip("-")
            cmd_arg[arg_name] = arg_value
        config.update_params(cmd_arg)

    args = config
    print(args.__dict__)

    # logger
    if "saves" in args.bert_model:
        log_dir = args.bert_model
        logger = Logger(log_dir)
        config = Config(main_conf_path=log_dir)
        old_args = copy.deepcopy(args)
        args.__dict__.update(config.__dict__)

        args.bert_model = old_args.bert_model
        args.do_train = old_args.do_train
        args.data_dir = old_args.data_dir
        args.task_name = old_args.task_name

        # apply system arguments if exist
        argv = sys.argv[1:]
        if len(argv) > 0:
            cmd_arg = OrderedDict()
            argvs = " ".join(sys.argv[1:]).split(" ")
            for i in range(0, len(argvs), 2):
                arg_name, arg_value = argvs[i], argvs[i + 1]
                arg_name = arg_name.strip("-")
                cmd_arg[arg_name] = arg_value
            config.update_params(cmd_arg)

    else:
        if not os.path.exists("saves"):
            os.mkdir("saves")
        log_dir = make_log_dir(os.path.join("saves", args.bert_model))
        logger = Logger(log_dir)
        config.save(log_dir)
    args.log_dir = log_dir

    # set CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # get dataset and processor
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    # build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = load_pretrained_model(args)

    ########### Training ###########

    # preprocessing
    orig_location = './data/orig/'
    preproc_location = './data/preproc/'
    save_data(input_location = orig_location, output_location = preproc_location)
    #extend_data(input_location = preproc_location, output_location = preproc_location)
    create_final_data(location=preproc_location)

    # Idiom
    if args.do_train and args.task_name == "idiom":
        train_dataloader = load_train_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        model, best_result = run_train(
            args,
            logger,
            model,
            train_dataloader,
            processor,
            task_name,
            label_list,
            tokenizer,
            output_mode,
        )

    # Load trained model
    if "saves" in args.bert_model:
        model = load_trained_model(args, model, tokenizer)



    ########### Inference ###########

    # idiom - eval
    if args.do_eval and task_name == "idiom":
       
        all_guids, eval_dataloader = load_eval_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode
        )
        final = run_dev(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=True) # return_preds=True 추가
        df_index = [i for i in range(len(final))]
        df = pd.DataFrame(final, index=df_index)
        df.to_csv('./submission/final.csv', index=False, encoding='cp949')

    logger.info(f"Saved to {logger.log_dir}")

    # create submission file
    create_submission()





def run_train(
    args,
    logger,
    model,
    train_dataloader,
    processor,
    task_name,
    label_list,
    tokenizer,
    output_mode,
    k=None,
):
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch

    # Prepare optimizer, scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.lr_schedule != False or args.lr_schedule.lower() != "none":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = { num_train_optimization_steps}")

    # Run training
    model.train()
    max_val_f1 = -1
    max_result = {}
    result_list = []
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            optimizer.zero_grad()
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            (
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                input_ids_2,
                input_mask_2,
                segment_ids_2,
                input_ids_3,
                input_mask_3,
                segment_ids_3,
                input_ids_4,
                input_mask_4,
                segment_ids_4,
            ) = batch

            # compute loss values
            logits = model(
                input_ids,
                input_ids_2,
                input_ids_3,
                input_ids_4,
                target_mask=(segment_ids == 1),
                target_mask_2=(segment_ids_2 == 1),
                target_mask_3=segment_ids_3,
                target_mask_4=segment_ids_4,
                attention_mask=input_mask,
                attention_mask_2=input_mask_2,
                attention_mask_3=input_mask_3,
                attention_mask_4=input_mask_4,
                token_type_ids=segment_ids,
                token_type_ids_2=segment_ids_2,
            )

            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

            # average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule != False or args.lr_schedule.lower() != "none":
                scheduler.step()

            
            tr_loss += loss.item()

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        # evaluate
        if args.do_dev:
            all_guids, dev_dataloader = load_dev_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode, k
            )
            result = run_dev(args, logger, model, dev_dataloader, all_guids, task_name)
            result_list.append(round(result["f1_macro"],4))
            # update
            if result["f1_macro"] > max_val_f1:
                max_val_f1 = result["f1_macro"]
                max_result = result
                save_model(args, model, tokenizer)
                best_model = copy.deepcopy(model)
                    
    print("result_list: ", result_list)
    logger.info(f"-----Best Result-----")
    for key in sorted(max_result.keys()):
        logger.info(f"  {key} = {str(max_result[key])}")

    return best_model, max_result


def run_dev(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=False):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    pred_guids = []
    out_label_ids = None

    for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)

        (
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            idx,
            input_ids_2,
            input_mask_2,
            segment_ids_2,
            input_ids_3,
            input_mask_3,
            segment_ids_3,
            input_ids_4,
            input_mask_4,
            segment_ids_4,
        ) = eval_batch


        with torch.no_grad():
            
            # compute loss values
            logits = model(
                input_ids,
                input_ids_2,
                input_ids_3,
                input_ids_4,
                target_mask=(segment_ids == 1),
                target_mask_2=(segment_ids_2 == 1),
                target_mask_3=segment_ids_3,
                target_mask_4=segment_ids_4,
                attention_mask=input_mask,
                attention_mask_2=input_mask_2,
                attention_mask_3=input_mask_3,
                attention_mask_4=input_mask_4,
                token_type_ids=segment_ids,
                token_type_ids_2=segment_ids_2,
                
            )
            loss_fct = nn.NLLLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                pred_guids.append([all_guids[i] for i in idx])
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                pred_guids[0].extend([all_guids[i] for i in idx])
                out_label_ids = np.append(
                    out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                )

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    # compute metrics
    result = compute_metrics(preds, out_label_ids)

    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

    if return_preds:
        return preds
    return result


def load_pretrained_model(args):
    # Pretrained Model
    bert = AutoModel.from_pretrained(args.bert_model)
    config = bert.config
    config.type_vocab_size = 2
    if "albert" in args.bert_model:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )
    else:
        bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
    bert._init_weights(bert.embeddings.token_type_embeddings)

    # Additional Layers

    model = AutoModelForSequenceClassification_SPV_MIP(
        args=args, Model=bert, config=config, num_labels=args.num_labels
    )

    model.to(args.device)
    if args.n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)
    return model


def save_model(args, model, tokenizer):
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.log_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.log_dir)

    # Good practice: save your training arguments together with the trained model
    output_args_file = os.path.join(args.log_dir, ARGS_NAME)
    torch.save(args, output_args_file)


def load_trained_model(args, model, tokenizer):
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)

    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(output_model_file), strict=False)
    else:
        model.load_state_dict(torch.load(output_model_file), strict=False)

    return model

def create_submission():

    final_pred = pd.read_csv('./submission/final.csv')
    submission= pd.read_csv('./submission/eval_submission_format.csv')


    submission['Label'] = final_pred['0']

    submission = submission.iloc[:len(final_pred),:]
    submission = submission.astype({'Label': float})
    submission = submission.astype({'Label': int})
    # submission['Setting'] = 'one_shot'
    submission.to_csv('./submission/task2_subtaska.csv', index=False, encoding='cp949')

if __name__ == "__main__":
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
