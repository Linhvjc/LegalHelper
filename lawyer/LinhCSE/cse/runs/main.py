from __future__ import annotations

import math
import os

import torch
from loguru import logger
from transformers import HfArgumentParser
from transformers import set_seed

import wandb
from cse.arguments import DatagArguments
from cse.arguments import ModelArguments
from cse.arguments import TrainingArguments
from cse.dataset.loader import CustomDataset
from cse.pipelines.trainers.CSEtrainer import CSETrainer
from cse.utils.io import load_file
from cse.utils.utils import generate_benchmark_filenames
from cse.utils.utils import load_tokenizer
from cse.utils.utils import MODEL_CLASSES
from cse.utils.utils import MODEL_PATH_MAP
from cse.utils.utils import print_recall_table


def main(data_args, model_args, train_args):
    wandb.init(
        project=train_args.wandb_project,
        name=train_args.wandb_run_name,
        config=vars(model_args),
    )

    set_seed(train_args.seed)

    # Pre Setup
    train_args.device_map = 'auto'
    train_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    train_args.ddp = train_args.world_size != 1
    if train_args.ddp:
        train_args.device_map = {'': int(os.environ.get('LOCAL_RANK') or 0)}
        train_args.gradient_accumulation_steps = (
            train_args.gradient_accumulation_steps // train_args.world_size
        )

    # Load tokenizer and model
    tokenizer = load_tokenizer(model_args)
    config_class, model_class, _ = MODEL_CLASSES[model_args.model_type]

    if model_args.pretrained:
        logger.info(
            f'Loading model from checkpoint {model_args.pretrained_path}',
        )
        model = model_class.from_pretrained(
            model_args.pretrained_path,
            torch_dtype=model_args.compute_dtype,
            device_map=train_args.device_map,
            args=model_args,
        )
    else:
        logger.info('Loading model ....')
        model_config = config_class.from_pretrained(
            model_args.model_name_or_path,
            finetuning_task=data_args.token_level,
        )
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=model_args.compute_dtype,
            config=model_config,
            device_map=train_args.device_map,
            args=model_args,
        )

    if model_args.resize_embedding_model:
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0)),
        )  # make the vocab size multiple of 8 # magic number

    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info(model)
    logger.info(model.dtype)
    logger.info(f'Vocab size: {len(tokenizer)}')

    # Load data
    training_file_path = os.path.join(
        data_args.data_dir,
        data_args.token_level,
        'train',
        'data.jsonl',
    )
    train_dataset = load_file(training_file_path)
    eval_file_path = os.path.join(
        data_args.data_dir, data_args.token_level, 'eval', 'data.jsonl',
    )
    eval_dataset = load_file(eval_file_path)
    train_dataset = CustomDataset(data_args, train_dataset, tokenizer)
    eval_dataset = CustomDataset(data_args, eval_dataset, tokenizer)

    trainer = CSETrainer(
        data_args=data_args,
        model_args=model_args,
        train_args=train_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    # import statistics

    # results = trainer.evaluate_on_benchmark()
    # for k, v in results.items():
    #     results[k] = statistics.mean(v)
    # print_recall_table(results)

    if train_args.do_train:
        trainer.train()


if __name__ == '__main__':
    parser = HfArgumentParser(
        (DatagArguments, ModelArguments, TrainingArguments),
    )
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    model_args.model_name_or_path = MODEL_PATH_MAP[model_args.model_type]
    # Check if parameter passed or if set within environ
    train_args.use_wandb = len(train_args.wandb_project) > 0 or (
        'WANDB_PROJECT' in os.environ and len(os.environ['WANDB_PROJECT']) > 0
    )

    # Only overwrite environ if wandb param passed
    if len(train_args.wandb_project) > 0:
        os.environ['WANDB_PROJECT'] = train_args.wandb_project
    if len(train_args.wandb_watch) > 0:
        os.environ['WANDB_WATCH'] = train_args.wandb_watch
    if len(train_args.wandb_log_model) > 0:
        os.environ['WANDB_LOG_MODEL'] = train_args.wandb_log_model

    data_args.benchmark_corpus_filenames = generate_benchmark_filenames(
        data_args.benchmark_dir,
    )
    main(data_args, model_args, train_args)
