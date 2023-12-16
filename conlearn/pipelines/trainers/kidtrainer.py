from __future__ import annotations

import statistics

import bitsandbytes as bnb
import numpy as np
import torch
import wandb
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import SequentialSampler
from tqdm.auto import tqdm
from tqdm.auto import trange
from transformers.optimization import AdamW
from transformers.optimization import get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import get_parameter_names

from .base import Trainer
from conlearn.encode.kid import KidEncoder
from conlearn.utils.io import load_file
from conlearn.utils.metric import recall
from conlearn.utils.regularize import EarlyStopping
from conlearn.utils.utils import log_embeddings_to_wandb


class KidTrainer(Trainer):
    def __init__(
        self,
        data_args,
        model_args,
        train_args,
        model: torch.nn.Module | None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        self.data_args = data_args
        self.model_args = model_args
        self.train_args = train_args

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = model

        self.tokenizer = tokenizer

        self.encoder = KidEncoder()

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.train_args.train_batch_size,
            drop_last=self.data_args.dataloader_drop_last,
            num_workers=self.data_args.dataloader_num_workers,
            pin_memory=self.data_args.dataloader_pin_memory,
        )

        if self.train_args.max_steps > 0:
            t_total = self.train_args.max_steps
            self.train_args.num_train_epochs = (
                self.train_args.max_steps
                // (len(train_dataloader) // self.train_args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // self.train_args.gradient_accumulation_steps
                * self.train_args.num_train_epochs
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = self.get_optimizer()

        scheduler = get_scheduler(
            self.train_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.train_args.warmup_steps,
            num_training_steps=t_total,
        )

        # Train!
        logger.info('***** Running training *****')
        logger.info(f'  Num examples = {len(self.train_dataset)}')
        logger.info(f'  Num Epochs = {self.train_args.num_train_epochs}')
        logger.info(
            f'  Total train batch size = {self.train_args.train_batch_size}',
        )
        logger.info(
            f'  Gradient Accumulation steps = {self.train_args.gradient_accumulation_steps}',
        )
        logger.info(f'  Total optimization steps = {t_total}')
        logger.info(f'  Logging steps = {self.train_args.logging_steps}')
        logger.info(f'  Save steps = {self.train_args.save_steps}')

        global_step = 0
        tr_loss = 0.0

        self.model.zero_grad()

        train_iterator = trange(
            int(self.train_args.num_train_epochs), desc='Epoch',
        )
        early_stopping = EarlyStopping(
            patience=self.train_args.early_stopping, verbose=True,
        )

        # Automatic Mixed Precision
        scaler = torch.cuda.amp.GradScaler()

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc='Iteration', position=0, leave=True,
            )
            logger.info(f'Epoch {_}')

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(
                    t.to(self.model.device)
                    for t in batch
                )  # GPU or CPU

                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'input_ids_positive': batch[2],
                    'attention_mask_positive': batch[3],
                    'is_train': True,
                }
                # outputs = self.model(**inputs)
                with torch.cuda.amp.autocast():
                    (
                        total_loss,
                        loss_ct,
                        _,
                        _,
                    ) = self.model(**inputs)

                wandb.log({'Train/Total Loss': total_loss.item()})
                wandb.log({'Train/Contrastive Loss': loss_ct.item()})

                if self.train_args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.train_args.gradient_accumulation_steps

                scaler.scale(total_loss).backward()

                tr_loss += total_loss.item()
                if (step + 1) % self.train_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_args.max_grad_norm,
                    )

                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()

                    self.model.zero_grad()
                    global_step += 1

                    lr = optimizer.param_groups[0]['lr']
                    wandb.log({'Train/learning_rate': lr})

                if (
                    self.train_args.logging_steps > 0
                    and global_step % self.train_args.logging_steps == 0
                ):
                    logger.info(
                        f'Tuning metrics: {self.train_args.tuning_metric}',
                    )
                    results = {}
                    results.update(self.evaluate_on_benchmark())
                    for k, v in results.items():
                        results[k] = statistics.mean(v)
                    results.update(self.evaluate())

                    wandb.log({'Eval/evaluate': results})
                    early_stopping(
                        results[self.train_args.tuning_metric],
                        self.model,
                        self.model_args.model_dir,
                        self.train_args.tuning_metric,
                    )
                    if early_stopping.early_stop:
                        logger.info('Early stopping')
                        break

                if 0 < self.train_args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.train_args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break
        return

    def evaluate(self):
        dataset = self.eval_dataset
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset,
            sampler=eval_sampler,
            batch_size=self.train_args.eval_batch_size,
            drop_last=self.data_args.dataloader_drop_last,
            num_workers=self.data_args.dataloader_num_workers,
            pin_memory=self.data_args.dataloader_pin_memory,
        )

        logger.info('***** Running evaluation on eval dataset *****')
        logger.info(f'  Num examples = {len(dataset)}')
        logger.info(f'  Batch size = {self.train_args.eval_batch_size}')

        eval_loss = 0.0
        eval_ct_loss = 0.0

        nb_eval_steps = 0

        self.model.eval()

        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                batch = tuple(
                    t.to(self.model.device)
                    for t in batch
                )  # GPU or CPU

                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'input_ids_positive': batch[2],
                    'attention_mask_positive': batch[3],
                    'is_train': True,
                }

                (total_loss, loss_ct, _, _) = self.model(**inputs)

                eval_loss += total_loss.item()
                eval_ct_loss += loss_ct.item()
            nb_eval_steps += 1

        eval_loss /= nb_eval_steps
        eval_ct_loss /= nb_eval_steps

        return {'total_loss': eval_loss, 'contrastive_loss': eval_ct_loss}

    def evaluate_on_benchmark(
        self,
        top_k_results: list[int] = [5, 10, 20],
        log_to_wandb: bool = False,
    ):
        """
        Evaluate the performance of the model on a benchmark dataset.
        Args:
            query_and_ground_truth (List[dict]): A list of dictionaries containing query and ground truth pairs.
                Each dictionary has the following keys:
                    - 'query': The query string.
                    - 'gt': The ground truth identifier.
            corpus (List[dict]): A list of dictionaries representing the corpus.
                Each dictionary has the following keys:
                    - 'text': The text content of the document.
                    - 'meta': A dictionary containing metadata information with the following keys:
                        - 'id': The identifier of the document.
                        - 'title': The title of the document.
                        - 'grade': The grade level of the document.
                        - 'unit': The unit of the document.
                        - 'section': The section of the document.
        """

        results = {}

        self.model.eval()

        for paths in self.data_args.benchmark_corpus_filenames:
            (benchmark_path, corpus_path) = paths
            name_benchmark = benchmark_path.split('/')[-1].split('.')[0]
            benchmark = load_file(benchmark_path).to_list()
            corpus = load_file(corpus_path).to_list()
            logger.info(f'Benchmark name: {name_benchmark}')

            embedding_corpus = None
            ids_corpus = []

            for i in range(0, len(corpus), self.train_args.eval_batch_size):
                documents = []
                for doc in corpus[i: i + self.train_args.eval_batch_size]:
                    ids_corpus.append(doc['meta']['id'])
                    documents.append(doc['text'])

                embedding = self.encoder.encode(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    texts=documents,
                    max_seq_len=self.data_args.max_seq_len_document,
                )

                if embedding_corpus is None:
                    embedding_corpus = embedding.detach().cpu().numpy()
                else:
                    embedding_corpus = np.append(
                        embedding_corpus,
                        embedding.detach().cpu().numpy(),
                        axis=0,
                    )

            if log_to_wandb:
                name = f'corpus_embeddings_{name_benchmark}'
                log_embeddings_to_wandb(embedding_corpus, name)

            embedding_query = None
            ground_truths = []

            for i in range(0, len(benchmark), self.train_args.eval_batch_size):
                queries = []
                for query in benchmark[i: i + self.train_args.eval_batch_size]:
                    ground_truths.append(query['gt'])
                    queries.append(query['query'])

                embedding = self.encoder.encode(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    texts=queries,
                    max_seq_len=self.data_args.max_seq_len_query,
                )

                if embedding_query is None:
                    embedding_query = embedding.detach().cpu().numpy()
                else:
                    embedding_query = np.append(
                        embedding_query,
                        embedding.detach().cpu().numpy(),
                        axis=0,
                    )

            if log_to_wandb:
                name = f'query_embeddings_{name_benchmark}'
                log_embeddings_to_wandb(embedding_query, name)

            # scores = embedding_query @ embedding_corpus.T
            # log_scores_similarity_to_wandb(scores)

            # Normalize the query and corpus embeddings
            embedding_query_norm = embedding_query / \
                np.linalg.norm(embedding_query, axis=1, keepdims=True)
            embedding_corpus_norm = embedding_corpus / \
                np.linalg.norm(embedding_corpus, axis=1, keepdims=True)

            # Compute the cosine similarity
            scores = np.dot(embedding_query_norm, embedding_corpus_norm.T)

            # embedding_query = torch.tensor(embedding_query)
            # embedding_corpus = torch.tensor(embedding_corpus)
            # import torch.nn.functional as F
            # scores = F.cosine_similarity(embedding_query.unsqueeze(0), embedding_corpus.unsqueeze(1), dim=-1)

            for score, ground_truth in zip(scores, ground_truths):
                for k in top_k_results:
                    if f'recall_{name_benchmark}_{k}' not in results:
                        results[f'recall_{name_benchmark}_{k}'] = []

                    ind = np.argpartition(score, -k)[-k:]
                    pred = list(map(ids_corpus.__getitem__, ind))

                    results[f'recall_{name_benchmark}_{k}'].append(
                        recall(pred, ground_truth),
                    )
        return results

    def get_optimizer(self):
        decay_parameters = get_parameter_names(
            self.model, [torch.nn.LayerNorm],
        )
        decay_parameters = [
            name for name in decay_parameters if 'bias' not in name
        ]
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if n in decay_parameters],
                'weight_decay': self.train_args.weight_decay,
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters() if n not in decay_parameters
                ],
                'weight_decay': 0.0,
            },
        ]
        if self.train_args.optimizer == 'AdamW':
            optimizer = AdamW(
                optimizer_grouped_parameters,
                betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
                lr=self.train_args.learning_rate,
                eps=self.train_args.adam_epsilon,
            )
        elif self.train_args.optimizer == '8bitAdam':
            optimizer = bnb.optim.Adam8bit(
                optimizer_grouped_parameters,
                betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
                lr=self.train_args.learning_rate,
                eps=self.train_args.adam_epsilon,
            )
        else:
            raise NotImplementedError(
                "Support is currently available only for the Adam optimizer.'",
            )
        return optimizer
