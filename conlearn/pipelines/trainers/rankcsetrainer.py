from __future__ import annotations

import copy
import os

import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm.auto import tqdm
from tqdm.auto import trange
from transformers.optimization import get_scheduler

from .kidtrainer import KidTrainer
from conlearn.models.modules.similarity import SimilarityFunctionWithTorch
from conlearn.models.rankcse_teacher import Teacher


class CLTrainer(KidTrainer):
    def __init__(
        self,
        data_collator,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_collator = data_collator
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.train_args.per_device_train_batch_size,
            drop_last=True,
            collate_fn=self.data_collator,
            pin_memory=True,
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

        optimizer = self.get_optimizer()

        scheduler = get_scheduler(
            self.train_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.train_args.warmup_steps,
            num_training_steps=t_total,
        )

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

        # self.state = TrainerState()

        # ! Teacher initial
        teacher = None
        if self.model_args.second_teacher_name_or_path is None:
            teacher_pooler = self.model_args.pooler_first_teacher
            teacher = Teacher(
                model_name_or_path=self.model_args.first_teacher_name_or_path,
                pooler=teacher_pooler,
            )
        else:
            first_pooler = self.model_args.pooler_first_teacher
            first_teacher = Teacher(
                model_name_or_path=self.model_args.first_teacher_name_or_path,
                pooler=first_pooler,
            )
            second_pooler = self.model_args.pooler_second_teacher
            second_teacher = Teacher(
                model_name_or_path=self.model_args.second_teacher_name_or_path,
                pooler=second_pooler,
            )

        global_step = 0
        tr_loss = torch.tensor(0.0).to(self.device)
        self.scaler = torch.cuda.amp.GradScaler()
        self.model.zero_grad()

        train_iterator = trange(
            int(self.train_args.num_train_epochs), desc='Epoch',
        )

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc='Iteration', position=0, leave=True,
            )
            logger.info(f'Epoch {_}')

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch.values())

                inputs = {
                    'input_ids': batch[0],
                    'token_type_ids': batch[1],
                    'attention_mask': batch[2],
                }

                input_ids = inputs['input_ids']
                token_type_ids = inputs['token_type_ids']
                attention_mask = inputs['attention_mask']

                with torch.cuda.amp.autocast():
                    batch_size = input_ids.shape[0]
                    num_sent = input_ids.shape[1]

                    # Flatten input for encoding by the teacher - (bsz * num_sent, len)
                    input_ids = input_ids.view((-1, input_ids.shape[-1]))
                    token_type_ids = token_type_ids.view(
                        (-1, token_type_ids.shape[-1]),
                    )
                    attention_mask = attention_mask.view(
                        (-1, attention_mask.shape[-1]),
                    )

                    teacher_inputs = copy.deepcopy(inputs)
                    teacher_inputs['input_ids'] = input_ids
                    teacher_inputs['attention_mask'] = attention_mask
                    teacher_inputs['token_type_ids'] = token_type_ids

                    cos = SimilarityFunctionWithTorch()
                    if teacher is not None:
                        # Single teacher
                        embeddings = teacher.encode(teacher_inputs)
                        embeddings = embeddings.view(
                            (batch_size, num_sent, -1),
                        )
                        z1T, z2T = embeddings[:, 0], embeddings[:, 1]

                        if self.train_args.fp16:
                            z1T = z1T.to(torch.float16)
                            z2T = z2T.to(torch.float16)

                        teacher_top1_sim_pred = (
                            cos(z1T, z2T) / self.model_args.tau2
                        )
                        inputs['super_teacher'] = teacher_top1_sim_pred
                        inputs['teacher_top1_sim_pred'] = teacher_top1_sim_pred

                    else:
                        # Weighted average of two teachers
                        embeddings1 = first_teacher.encode(teacher_inputs)
                        embeddings2 = second_teacher.encode(teacher_inputs)
                        embeddings1 = embeddings1.view(
                            (batch_size, num_sent, -1),
                        )
                        embeddings2 = embeddings2.view(
                            (batch_size, num_sent, -1),
                        )
                        first_teacher_z1, first_teacher_z2 = embeddings1[
                            :,
                            0,
                        ], embeddings1[:, 1]
                        second_teacher_z1, second_teacher_z2 = embeddings2[
                            :,
                            0,
                        ], embeddings2[:, 1]

                        if self.train_args.fp16:
                            first_teacher_z1 = first_teacher_z1.to(
                                torch.float16,
                            )
                            first_teacher_z2 = first_teacher_z2.to(
                                torch.float16,
                            )
                            second_teacher_z1 = second_teacher_z1.to(
                                torch.float16,
                            )
                            second_teacher_z2 = second_teacher_z2.to(
                                torch.float16,
                            )

                        first_teacher_top1_sim = (
                            cos(first_teacher_z1, first_teacher_z2) /
                            self.model_args.tau2
                        )
                        second_teacher_top1_sim = (
                            cos(second_teacher_z1, second_teacher_z2) /
                            self.model_args.tau2
                        )
                        teacher_top1_sim_pred = (self.model_args.alpha_ * first_teacher_top1_sim) + (
                            (1.0 - self.model_args.alpha_) *
                            second_teacher_top1_sim
                        )
                        inputs['teacher_top1_sim_pred'] = teacher_top1_sim_pred
                        inputs['super_teacher'] = first_teacher_top1_sim
                    inputs = {
                        key: value.to(self.device)
                        for key, value in inputs.items()
                    }

                    self.model = self.model.to(self.device)
                    outputs = self.model(**inputs)
                    total_loss = outputs.loss

                if self.train_args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.train_args.gradient_accumulation_steps

                self.scaler.scale(total_loss).backward()
                tr_loss += total_loss.item()

                if (step + 1) % self.train_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_args.max_grad_norm,
                    )

                    self.scaler.step(optimizer)
                    scheduler.step()
                    self.scaler.update()

                    self.model.zero_grad()
                    global_step += 1

        return tr_loss

    def save_model(self, output_dir: str | None = None):
        output_dir = output_dir or self.train_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f'Save model sucessfully at {output_dir}')


class CLMocoTrainer(CLTrainer):
    def __init__(self, data_collator, **kwargs):
        super().__init__(data_collator, **kwargs)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.train_args.per_device_train_batch_size,
            drop_last=True,
            collate_fn=self.data_collator,
            pin_memory=True,
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

        optimizer = self.get_optimizer()

        scheduler = get_scheduler(
            self.train_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.train_args.warmup_steps,
            num_training_steps=t_total,
        )

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

        # self.state = TrainerState()

        # ! Teacher initial
        teacher = None
        if self.model_args.second_teacher_name_or_path is None:
            teacher_pooler = self.model_args.pooler_first_teacher
            teacher = Teacher(
                model_name_or_path=self.model_args.first_teacher_name_or_path,
                pooler=teacher_pooler,
            )
        else:
            first_pooler = self.model_args.pooler_first_teacher
            first_teacher = Teacher(
                model_name_or_path=self.model_args.first_teacher_name_or_path,
                pooler=first_pooler,
            )
            second_pooler = self.model_args.pooler_second_teacher
            second_teacher = Teacher(
                model_name_or_path=self.model_args.second_teacher_name_or_path,
                pooler=second_pooler,
            )

        global_step = 0
        tr_loss = torch.tensor(0.0).to(self.device)
        self.scaler = torch.cuda.amp.GradScaler()
        self.model.zero_grad()

        train_iterator = trange(
            int(self.train_args.num_train_epochs), desc='Epoch',
        )

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc='Iteration', position=0, leave=True,
            )
            logger.info(f'Epoch {_}')

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch.values())

                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[2],
                }

                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                token_type_ids = batch[1]

                with torch.cuda.amp.autocast():
                    batch_size = input_ids.shape[0]
                    num_sent = input_ids.shape[1]

                    # Flatten input for encoding by the teacher - (bsz * num_sent, len)
                    input_ids = input_ids.view((-1, input_ids.shape[-1]))
                    token_type_ids = token_type_ids.view(
                        (-1, token_type_ids.shape[-1]),
                    )
                    attention_mask = attention_mask.view(
                        (-1, attention_mask.shape[-1]),
                    )

                    teacher_inputs = copy.deepcopy(inputs)
                    teacher_inputs['input_ids'] = input_ids
                    teacher_inputs['attention_mask'] = attention_mask
                    teacher_inputs['token_type_ids'] = token_type_ids

                    cos = SimilarityFunctionWithTorch()
                    if teacher is not None:
                        # Single teacher
                        embeddings = teacher.encode(teacher_inputs)
                        embeddings = embeddings.view(
                            (batch_size, num_sent, -1),
                        )
                        z1T, z2T = embeddings[:, 0], embeddings[:, 1]

                        if self.train_args.fp16:
                            z1T = z1T.to(torch.float16)
                            z2T = z2T.to(torch.float16)

                        teacher_top1_sim_pred = (
                            cos(z1T, z2T) / self.model_args.tau2
                        )
                        inputs['teacher_pred'] = teacher_top1_sim_pred

                    else:
                        # Weighted average of two teachers
                        embeddings1 = first_teacher.encode(teacher_inputs)
                        embeddings2 = second_teacher.encode(teacher_inputs)
                        embeddings1 = embeddings1.view(
                            (batch_size, num_sent, -1),
                        )
                        embeddings2 = embeddings2.view(
                            (batch_size, num_sent, -1),
                        )
                        first_teacher_z1, first_teacher_z2 = embeddings1[
                            :,
                            0,
                        ], embeddings1[:, 1]
                        second_teacher_z1, second_teacher_z2 = embeddings2[
                            :,
                            0,
                        ], embeddings2[:, 1]

                        if self.train_args.fp16:
                            first_teacher_z1 = first_teacher_z1.to(
                                torch.float16,
                            )
                            first_teacher_z2 = first_teacher_z2.to(
                                torch.float16,
                            )
                            second_teacher_z1 = second_teacher_z1.to(
                                torch.float16,
                            )
                            second_teacher_z2 = second_teacher_z2.to(
                                torch.float16,
                            )

                        first_teacher_top1_sim = (
                            cos(first_teacher_z1, first_teacher_z2) /
                            self.model_args.tau2
                        )
                        second_teacher_top1_sim = (
                            cos(second_teacher_z1, second_teacher_z2) /
                            self.model_args.tau2
                        )
                        teacher_top1_sim_pred = (self.model_args.alpha_ * first_teacher_top1_sim) + (
                            (1.0 - self.model_args.alpha_) *
                            second_teacher_top1_sim
                        )
                        inputs['teacher_pred'] = teacher_top1_sim_pred
                    inputs = {
                        key: value.to(self.device)
                        for key, value in inputs.items()
                    }

                    self.model = self.model.to(self.device)
                    outputs = self.model(**inputs)
                    total_loss = outputs.loss

                if self.train_args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.train_args.gradient_accumulation_steps

                self.scaler.scale(total_loss).backward()
                tr_loss += total_loss.item()

                if (step + 1) % self.train_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_args.max_grad_norm,
                    )

                    self.scaler.step(optimizer)
                    scheduler.step()
                    self.scaler.update()

                    self.model.zero_grad()
                    global_step += 1

        return tr_loss

    def save_model(self, output_dir: str | None = None):
        output_dir = output_dir or self.train_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f'Save model sucessfully at {output_dir}')
