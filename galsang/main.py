import random
import copy
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from utils import Config, Logger, make_log_dir
from modeling import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification_SPV,
    AutoModelForSequenceClassification_MIP,
    AutoModelForSequenceClassification_SPV_MIP,
)
from run_classifier_dataset_utils import processors, output_modes, compute_metrics
from data_loader import load_train_data, load_eval_data, load_dev_data
from data_preproc import *

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"


def main():
    # read configs
    config = Config(main_conf_path="./")

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
    save_data(input_location=orig_location, output_location=preproc_location)
    # extend_data(input_location = preproc_location, output_location = preproc_location)
    create_final_data(location=preproc_location)

    if args.do_train:
        train_dataloader = load_train_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode)
        model, best_result = run_train(
            args, logger, model, train_dataloader, processor, task_name, label_list, tokenizer, output_mode)
    # Load trained model
    if "saves" in args.bert_model:
        model = load_trained_model(args, model, tokenizer)

    ########### Inference ###########
    if args.do_dev:
        all_guids, dev_dataloader = load_dev_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode)
        run_dev(args, logger, model, dev_dataloader, all_guids, task_name)

    if args.do_eval:
        all_guids, eval_dataloader = load_eval_data(
            args, logger, processor, task_name, label_list, tokenizer, output_mode)
        final = run_dev(args, logger, model, eval_dataloader, all_guids, task_name, return_preds=True)
        df_index = [i for i in range(len(final))]
        df = pd.DataFrame(final, index=df_index)
        df.to_csv('./submission/final.csv', index=False, encoding='cp949')

    # create submission file
    create_submission()


def run_train(args, logger, model, train_dataloader, processor, task_name, label_list, tokenizer, output_mode):
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
    logger.info(f"  Num steps = {num_train_optimization_steps}")

    # Run training
    model.train()
    max_result = {'f1_macro': -1}
    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(train_dataloader):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            if args.model_type in ["MELBERT_MIP", "MELBERT", "MELBERT_CONTEXT"]:
                (input_ids, input_mask, segment_ids, label_ids,
                 input_ids_2, input_mask_2, segment_ids_2) = batch
            else:
                input_ids, input_mask, segment_ids, label_ids = batch

            # compute loss values
            if args.model_type in ["BERT_SEQ", "BERT_BASE", "MELBERT_SPV", "MELBERT_SPV_CONTEXT"]:
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
            elif args.model_type in ["MELBERT_MIP", "MELBERT", "MELBERT_CONTEXT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

            # average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule != False or args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch + 1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")

        # evaluate
        if args.do_dev:
            all_guids, dev_dataloader = load_dev_data(
                args, logger, processor, task_name, label_list, tokenizer, output_mode)
            result = run_dev(args, logger, model, dev_dataloader, all_guids, task_name)

            # update
            if result['f1_macro'] > max_result['f1_macro']:
                max_result = result
                save_model(args, model, tokenizer)
                best_model = copy.deepcopy(model)

    logger.info(f"-----Best Result (Training)-----")
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

    # for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
    for eval_batch in eval_dataloader:
        eval_batch = tuple(t.to(args.device) for t in eval_batch)

        if args.model_type in ["MELBERT_MIP", "MELBERT", "MELBERT_CONTEXT"]:
            (input_ids, input_mask, segment_ids, label_ids, idx,
             input_ids_2, input_mask_2, segment_ids_2) = eval_batch
        else:
            input_ids, input_mask, segment_ids, label_ids, idx = eval_batch

        with torch.no_grad():
            # compute loss values
            if args.model_type in ["BERT_BASE", "BERT_SEQ", "MELBERT_SPV", "MELBERT_SPV_CONTEXT"]:
                logits = model(
                    input_ids,
                    target_mask=(segment_ids == 1),
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
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
                    out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

            elif args.model_type in ["MELBERT_MIP", "MELBERT", "MELBERT_CONTEXT"]:
                logits = model(
                    input_ids,
                    input_ids_2,
                    target_mask=(segment_ids == 1),
                    target_mask_2=segment_ids_2,
                    attention_mask_2=input_mask_2,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
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
                    out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

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
    config.type_vocab_size = 3
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
    if args.model_type in ["BERT_BASE"]:
        model = AutoModelForSequenceClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "BERT_SEQ":
        model = AutoModelForTokenClassification(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type in ["MELBERT_SPV", "MELBERT_SPV_CONTEXT"]:
        model = AutoModelForSequenceClassification_SPV(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type == "MELBERT_MIP":
        model = AutoModelForSequenceClassification_MIP(
            args=args, Model=bert, config=config, num_labels=args.num_labels
        )
    if args.model_type in ["MELBERT", "MELBERT_CONTEXT"]:
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
        model.module.load_state_dict(torch.load(output_model_file))
    else:
        model.load_state_dict(torch.load(output_model_file))

    return model


def create_submission():
    final_pred = pd.read_csv('./submission/final.csv')
    submission = pd.read_csv('./submission/eval_submission_format.csv')

    submission['Label'] = final_pred['0']

    submission = submission.iloc[:len(final_pred), :]
    submission = submission.astype({'Label': float})
    submission = submission.astype({'Label': int})

    submission.to_csv('./submission/task2_subtaska.csv', index=False, encoding='cp949')


if __name__ == "__main__":
    main()
