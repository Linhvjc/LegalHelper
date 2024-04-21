import os
import argparse
import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer

from utils import set_seed, logger
from components.eval import BenchmarkEvaluator
from components.models import NLIFinetuneModel
from components.datasets import NLIDataset


def main(args):
    local_rank = args.local_rank
    logger.info(f"local_rank: {local_rank}")

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:' + str(local_rank))
    set_seed(args.seed)

    logger.info("Load Model")
    bert_model = NLIFinetuneModel(
        model_path_or_name=args.model_base,
        temp=args.temp,
        queue_len=args.queue_len,
        pooler_type=args.pooler_type)
    if args.model_path:
        bert_model.student_model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    bert_model = bert_model.to(device)
    bert_model.train()

    logger.info("Load NLI Dataset")
    dataset = NLIDataset(data_path=args.data_path,
                         tokenizer=tokenizer, use_vi_tokenizer=args.use_vi_tokenizer,
                         max_seq_len=args.max_seq_len)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=10, shuffle=True, drop_last=True)

    logger.info("Load Benchmark Evaluator")
    evaluator = BenchmarkEvaluator(use_vi_tokenizer=args.use_vi_tokenizer)
    metric_names = ["history_recall@1", "vinli_recall@1"]

    queue = torch.tensor([])
    num_epoch = args.epochs
    optim = AdamW(bert_model.parameters(), lr=args.lr, eps=1e-8, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=0.1 * len(dataloader),
        num_training_steps=len(dataloader) * num_epoch)

    steps = 0
    scaler = GradScaler()
    max_norm = 1.0
    max_result = 0
    logger.info("Trainer Starting")
    for epoch in range(num_epoch):
        logger.info(f"Epoch: {epoch}")
        logger.info(f"Time: {datetime.datetime.now()}")

        for sample_zh in tqdm(dataloader):
            for key in sample_zh:
                sample_zh[key] = sample_zh[key].reshape(-1, args.max_seq_len)
                sample_zh[key] = sample_zh[key].to(device)

            optim.zero_grad()
            with autocast():
                loss, queue, temp = bert_model(queue=queue.to(device), **sample_zh)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm)
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            # evaluate
            if steps % args.eval_step == 0:
                logger.info(f"===> Evaluator Step: {steps // args.eval_step} <====")
                all_results = evaluator.get_result(
                    model=bert_model.student_model,
                    tokenizer=tokenizer,
                    pooler_type=args.pooler_type,
                    device=device)
                results = sum([all_results[metric] for metric in metric_names]) / len(metric_names)
                bert_model.train()

                if os.path.exists(args.save_model_path) == False:
                    os.system('mkdir ' + args.save_model_path)

                # save model
                if results > max_result:
                    max_result = results
                    logger.info(f"Step: {steps} / Model Saving =====> Current Evaluation Results: {results}")
                    now_save = 'eval_step_{}-epoch_{}-pooler_{}-seed_{}'.format(
                        steps, args.epochs, args.pooler_type, args.seed)
                    os.system('rm ' + args.save_model_path + '/*.pth')
                    torch.save(bert_model.student_model.state_dict(),
                               args.save_model_path + '/' + now_save + '.pth'.format(steps))
                else:
                    logger.info(f"Step: {steps} / Best Results: {max_result} / Current Evaluation Results: {results}")
            steps += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Semantic Textual Similarity Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--data_path', type=str, default='./data/labeled/SimCSE/nli_for_simcse.csv', help='data_path')
    parser.add_argument('--model_base', type=str, default='vinai/phobert-base-v2', help='model_base')
    parser.add_argument('--model_path', type=str,
                        default="./models/experiments/exp1/eval_step_2375-num_hidden_12-epoch_10-pooler_cls-seed_1111.pth",
                        help='model_path')
    parser.add_argument('--lr', type=float, default=2e-5, help='lr')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--save_model_path', type=str, default="./models/experiments/exp2", help='save_model_path')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--eval_step', type=int, default=125, help='eval_step')
    parser.add_argument('--temp', type=int, default=20, help='temp')
    parser.add_argument('--queue_len', type=int, default=3, help='queue_len')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max_seq_len')
    parser.add_argument('--student_eval', type=int, default=0, help='eval')
    parser.add_argument('--pooler_type', type=str, default='cls', choices=['cls', 'cbp'], help='pooler_type')
    parser.add_argument('--use_vi_tokenizer', type=bool, default=True, help='use_vi_tokenizer')
    parser.add_argument('--seed', type=int, default=1111, help='seed')
    args = parser.parse_args()

    # set device
    torch.cuda.set_device(args.local_rank)
    main(args)
