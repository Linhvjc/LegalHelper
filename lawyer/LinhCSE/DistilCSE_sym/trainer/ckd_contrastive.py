import os
from tqdm import tqdm
import argparse
import datetime
import json
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig

from utils import set_seed, logger
from components.models import ContrastiveKD
from components.datasets import OnlineDataset, OnlineDatasetAsym
from components.eval import BenchmarkEvaluator
from activations.swiglu import SwiGLURoberta
import wandb


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Textual Similarity Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--data_path', type=str, default="./data/corpus/SS_Sym_VA_train_193M.txt", help='data path')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max_seq_len')
    parser.add_argument('--save_model_path', type=str, default='./models/experiments/exp1', help='save_model_path')
    parser.add_argument('--student_model_path_or_name', type=str, default="vinai/phobert-base-v2",
                        help='student_model_path')
    parser.add_argument('--teacher_model_path_or_name', type=str,
                        default="./models/in_model/SS_Sym_VA_CTMethod_T1.2024", help='teacher_model_path')
    parser.add_argument('--eval_steps', type=int, default=125, help='eval_steps')
    parser.add_argument('--queue_len', type=int, default=50000, help='queue_len')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--num_workers', default=8, type=int, help='num_workers')
    parser.add_argument('--lr', default=2e-4, type=float, help='lr')
    parser.add_argument('--temp', default=1.0, type=float, help='temp')
    parser.add_argument('--temp_exp', default=1, type=int, help='temp_exp')
    parser.add_argument('--early_stop', default=3, type=int, help='early_stop')
    parser.add_argument('--pooler_type', type=str, default='cls', choices=['cls', 'cbp'], help='pooler_type')
    parser.add_argument('--use_vi_tokenizer', type=bool, default=True, help='use_vi_tokenizer')
    parser.add_argument('--seed', default=1111, type=int, help='seed')
    parser.add_argument('--mse', default=0, type=int, help='mse')

    return parser


def load_models(args):
    logger.info("Load Teacher Model")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path_or_name)
    teacher_model = AutoModel.from_pretrained(args.teacher_model_path_or_name)

    logger.info("Load Student Model")
    last_model_name = args.student_model_path_or_name
    config = AutoConfig.from_pretrained(last_model_name)
    student_model = AutoModel.from_pretrained(last_model_name, config=config)
    student_tokenizer = AutoTokenizer.from_pretrained(last_model_name)

    # for layer in student_model.encoder.layer:
    #     layer.intermediate = SwiGLURoberta(config)

    return teacher_tokenizer, student_tokenizer, teacher_model, student_model


def main(args):
    wandb.init(
        project='distill-cse',
        name='Asym',
        config=vars(args),
    )
    logger.info("All Parameters:")
    logger.info(json.dumps(args.__dict__, indent=2, ensure_ascii=False, sort_keys=True))

    local_rank = args.local_rank
    logger.info(f"local_rank: {local_rank}")

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:' + str(local_rank))
    set_seed(args.seed)

    teacher_tokenizer, student_tokenizer, teacher_model, student_model = load_models(args)

    logger.info("Load Online Dataset")
    # dataset = OnlineDataset(
    #     text_path=args.data_path,
    #     student_tokenizer=deepcopy(student_tokenizer),
    #     teacher_tokenizer=deepcopy(teacher_tokenizer),
    #     max_seq_len=args.max_seq_len,
    #     use_vi_tokenizer=args.use_vi_tokenizer)
    
    dataset = OnlineDatasetAsym(
        text_path=args.data_path,
        student_tokenizer=deepcopy(student_tokenizer),
        teacher_tokenizer=deepcopy(teacher_tokenizer),
        max_query_len=128,
        max_doc_len=128,
        use_vi_tokenizer=args.use_vi_tokenizer
    )
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=True, drop_last=True)

    logger.info("Load Benchmark Evaluator")
    evaluator = BenchmarkEvaluator(use_vi_tokenizer=args.use_vi_tokenizer)
    metric_names = ["history_recall@1", "vinli_recall@1"]

    model = ContrastiveKD(student_model=student_model, teacher_model=teacher_model, args=args)
    model = model.to(device)
    model.train()
    queue_query = torch.Tensor([])
    queue_document = torch.Tensor([])

    scaler = GradScaler()
    optim = AdamW(model.parameters(), lr=args.lr, eps=1e-8, betas=(0.9, 0.98))
    num_epoch = args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=0.1 * len(dataloader), num_training_steps=len(dataloader) * num_epoch)

    steps = 0
    max_norm = 1.0
    max_result = 0
    pre_epoch = 0
    logger.info("Trainer Starting")
    for epoch in range(num_epoch):
        time1 = datetime.datetime.now()
        logger.info(f"Epoch: {epoch}")
        logger.info(f"Time: {time1}")

        if epoch - pre_epoch > args.early_stop:
            logger.info('epoch{}'.format(epoch))
            logger.info('epoch{}'.format(pre_epoch))
            os.system('bash clear_single.sh')
            break

        # for student_inputs, teacher_inputs in tqdm(dataloader):
        #     optim.zero_grad()
        #     with autocast():
        #         for k in student_inputs:
        #             student_inputs[k] = student_inputs[k].to(device)
        #         for k in teacher_inputs:
        #             teacher_inputs[k] = teacher_inputs[k].to(device)

        #         loss, queue, temp = model(student_inputs=student_inputs, 
        #                                   teacher_inputs=teacher_inputs,
        #                                   queue=queue, steps=steps,
        #                                   queue_len=args.queue_len, 
        #                                   mse=args.mse)
        for (student_inputs_query,
            student_inputs_document, 
            teacher_inputs_query, 
            teacher_inputs_document) in tqdm(dataloader):
            optim.zero_grad()
            with autocast():
                for k in student_inputs_query:
                    student_inputs_query[k] = student_inputs_query[k].to(device)
                for k in student_inputs_document:
                    student_inputs_document[k] = student_inputs_document[k].to(device)
                for k in teacher_inputs_query:
                    teacher_inputs_query[k] = teacher_inputs_query[k].to(device)
                for k in teacher_inputs_document:
                    teacher_inputs_document[k] = teacher_inputs_document[k].to(device)

                loss, queue_query, queue_document, temp = model(student_inputs_query=student_inputs_query, 
                                          student_inputs_document=student_inputs_document,
                                          teacher_inputs_query=teacher_inputs_query,
                                          teacher_inputs_document=teacher_inputs_document,
                                          queue_query=queue_query, 
                                          queue_document=queue_document, 
                                          steps=steps,
                                          queue_len=args.queue_len, 
                                          mse=args.mse)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            wandb.log({"Train/loss":loss})

            # evaluate
            if steps % args.eval_steps == 0:
                logger.info(f"===> Evaluator Step: {steps // args.eval_steps} <====")
                all_results = evaluator.get_result(
                    model=student_model,
                    tokenizer=student_tokenizer,
                    pooler_type=args.pooler_type,
                    device=device)
                wandb.log(all_results)
                results = sum([all_results[metric] for metric in metric_names]) / len(metric_names)
                student_model.train()

                if os.path.exists(args.save_model_path) == False:
                    os.system('mkdir ' + args.save_model_path)

                # save model
                if results > max_result:
                    pre_epoch = epoch
                    max_result = results
                    logger.info(f"Step: {steps} / Model Saving =====> Current Evaluation Results: {results}")
                    now_save = 'eval_step_{}-epoch_{}-pooler_{}-seed_{}'.format(
                        steps, args.epochs, args.pooler_type, args.seed)
                    os.system('rm ' + args.save_model_path + '/*.pth')
                    torch.save(student_model.state_dict(),
                               args.save_model_path + '/' + now_save + '.pth'.format(steps))
                else:
                    logger.info(f"Step: {steps} / Best Results: {max_result} / Current Evaluation Results: {results}")
            steps += 1


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # set device
    torch.cuda.set_device(args.local_rank)
    main(args)
