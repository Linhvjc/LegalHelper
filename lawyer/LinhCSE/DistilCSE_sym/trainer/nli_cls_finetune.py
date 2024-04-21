import os
import argparse
import datetime
from tqdm import tqdm
import json

from focal_loss.focal_loss import FocalLoss
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer

from utils import set_seed, logger
from components.eval import BenchmarkEvaluator
from components.models import NLI_CLS_Model
from components.datasets import NLI_CLS_Dataset
from components.losses import Poly1FocalLoss, Poly1CrossEntropyLoss


def main(args):
    logger.info("All Parameters:")
    logger.info(json.dumps(args.__dict__, indent=2, ensure_ascii=False, sort_keys=True))

    local_rank = args.local_rank
    logger.info(f"local_rank: {local_rank}")

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:' + str(local_rank))
    set_seed(args.seed)

    logger.info("Load NLI CLS Dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    dataset = NLI_CLS_Dataset(
        data_path=args.data_path,
        tokenizer=tokenizer, use_vi_tokenizer=args.use_vi_tokenizer,
        max_seq_len=args.max_seq_len)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=10, shuffle=True, drop_last=True)
    num_classes = dataset.get_num_classes()
    class_weights = dataset.compute_class_weights()
    logger.info(f"Num Classes: {dataset.get_num_classes()}")
    logger.info(f"Details Classes: {dataset.get_label_classes()}")

    logger.info("Load Model")
    bert_model = NLI_CLS_Model(
        model_path_or_name=args.model_base,
        num_labels=dataset.get_num_classes(),
        pooler_type=args.pooler_type)
    if args.model_path:
        bert_model.bert.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)
    bert_model = bert_model.to(device)
    bert_model.train()

    # loss criterion
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if args.loss_type == "focal_loss":
        criterion = FocalLoss(gamma=0.7, weights=class_weights)
    elif args.loss_type == "poly_focal":
        criterion = Poly1FocalLoss(num_classes=num_classes, reduction=args.poly_reduce_type)
    elif args.loss_type == "poly_cross_entropy":
        criterion = Poly1CrossEntropyLoss(num_classes=num_classes, reduction=args.poly_reduce_type)

    logger.info("Load Benchmark Evaluator")
    evaluator = BenchmarkEvaluator(use_vi_tokenizer=args.use_vi_tokenizer)
    metric_names = ["history_recall@1", "vinli_recall@1"]

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

        for batch in tqdm(dataloader):
            sample_zh = batch[0]
            labels = batch[1].to(device)
            for key in sample_zh:
                sample_zh[key] = sample_zh[key].to(device)

            optim.zero_grad()
            loss = 0.0
            with autocast():
                logits, probs = bert_model(**sample_zh)
                if args.loss_type == "cross_entropy":
                    loss = criterion(logits, labels)
                if args.loss_type == "focal_loss":
                    loss = criterion(probs, labels)
                elif args.loss_type == "poly_focal":
                    loss = criterion(logits, labels)
                elif args.loss_type == "poly_cross_entropy":
                    loss = criterion(logits, labels)

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
                    model=bert_model.bert,
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
                    torch.save(bert_model.bert.state_dict(),
                               args.save_model_path + '/' + now_save + '.pth'.format(steps))
                else:
                    logger.info(f"Step: {steps} / Best Results: {max_result} / Current Evaluation Results: {results}")
            steps += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Semantic Textual Similarity Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--data_path', type=str,
                        default='./data/labeled/VI_NLI/processed/UIT_ViNLI.json', help='data_path')
    parser.add_argument('--model_base', type=str, default='vinai/phobert-base-v2', help='model_base')
    parser.add_argument('--model_path', type=str,
                        default="./models/experiments/exp1/eval_step_2375-num_hidden_12-epoch_10-pooler_cls-seed_1111.pth",
                        help='model_path')
    parser.add_argument('--lr', type=float, default=2e-5, help='lr')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--save_model_path', type=str, default="./models/experiments/exp2", help='save_model_path')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--eval_step', type=int, default=125, help='eval_step')
    parser.add_argument('--max_seq_len', type=int, default=256, help='max_seq_len')
    parser.add_argument('--pooler_type', type=str, default='cls', choices=['cls', 'cbp'], help='pooler_type')
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'poly_focal', 'poly_cross_entropy'], help='loss_type')
    parser.add_argument('--poly_reduce_type', type=str, default='mean',
                        choices=['mean', 'none', 'sum'], help='poly_reduce_type')
    parser.add_argument('--use_vi_tokenizer', type=bool, default=True, help='use_vi_tokenizer')
    parser.add_argument('--seed', type=int, default=1111, help='seed')
    args = parser.parse_args()

    # set device
    torch.cuda.set_device(args.local_rank)
    main(args)
