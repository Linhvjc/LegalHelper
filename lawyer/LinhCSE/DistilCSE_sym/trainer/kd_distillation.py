import os
from copy import deepcopy
import datetime
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig
from info_nce import InfoNCE

from components.models import LinearBert
from components.datasets import OnlineDataset
from components.eval import BenchmarkEvaluator
from utils import set_seed, logger


def main(args):
    local_rank = args.local_rank
    logger.info(f"local_rank: {local_rank}")

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:' + str(local_rank))
    set_seed(args.seed)

    logger.info("Load Teacher Model")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path_or_name)
    teacher_model = AutoModel.from_pretrained(args.teacher_model_path_or_name)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    out_features = 1024
    try:
        out_features = teacher_model.pooler.dense.out_features
    except:
        logger.error(f"Not get out_features of teacher model")

    logger.info("Load Student Model")
    # dic_num2hidden = {
    #     12: "bert-base-uncased",
    #     6: "2nd_General_TinyBERT_6L_768D",
    #     4: "2nd_General_TinyBERT_4L_312D",
    # }
    # last_model_name = dic_num2hidden[args.num_hidden_layers]
    config = AutoConfig.from_pretrained(args.student_model_path_or_name)
    student_model = LinearBert(args.student_model_path_or_name,
                               config=config,
                               pooler_type=args.pooler_type,
                               out_features=out_features)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_path_or_name)
    student_model = student_model.to(device)
    student_model.train()

    # load dataset and evaluator
    logger.info("Load Online Dataset")
    dataset = OnlineDataset(
        text_path=args.data_path,
        student_tokenizer=deepcopy(student_tokenizer),
        teacher_tokenizer=deepcopy(teacher_tokenizer),
        max_seq_len=128,
        use_vi_tokenizer=args.use_vi_tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=True, drop_last=True)

    logger.info("Load Benchmark Evaluator")
    evaluator = BenchmarkEvaluator(use_vi_tokenizer=args.use_vi_tokenizer)
    metric_names = ["history_recall@1", "vinli_recall@1"]

    criterion = nn.MSELoss()
    if args.loss_type == "infonce":
        criterion = InfoNCE()
    scaler = GradScaler()
    num_epoch = args.epochs
    optim = AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=args.lr, eps=1e-8, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=0.1 * len(dataloader),
        num_training_steps=len(dataloader) * num_epoch)

    max_result = 0
    pre_epoch = 0
    steps = 0
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

        for student_inputs, teacher_inputs in tqdm(dataloader, f"Epoch: {epoch}"):
            for k in student_inputs:
                student_inputs[k] = student_inputs[k].to(device)
            for k in teacher_inputs:
                teacher_inputs[k] = teacher_inputs[k].to(device)
            optim.zero_grad()
            with autocast():
                student_features = student_model(**student_inputs, output_hidden_states=True, return_dict=True)
                with torch.no_grad():
                    if 'cls' in args.pooler_type:
                        teacher_features = teacher_model(
                            **teacher_inputs,
                            output_hidden_states=True,
                            return_dict=True).pooler_output
                    elif 'cbp' in args.pooler_type:
                        teacher_features = teacher_model(
                            **teacher_inputs,
                            output_hidden_states=True,
                            return_dict=True).last_hidden_state[:, 0]

                loss = criterion(student_features, teacher_features)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            # evaluator
            if steps % args.eval_step == 0:
                logger.info(f"===> Evaluator Step: {steps // args.eval_step} <====")
                all_results = evaluator.get_result(
                    model=student_model.bert,
                    tokenizer=student_tokenizer,
                    pooler_type=args.pooler_type,
                    device=device)
                results = sum([all_results[metric] for metric in metric_names]) / len(metric_names)
                student_model.train()

                if os.path.exists(args.save_model_path) == False:
                    os.system('mkdir ' + args.save_model_path)
                # save model
                if results > max_result:
                    pre_epoch = epoch
                    max_result = results
                    logger.info(f"Step: {steps} / Model Saving =====> Current Evaluation Results: {results}")
                    now_save = 'eval_step_{}-num_hidden_{}-epoch_{}-pooler_{}-seed_{}'.format(
                        steps, args.num_hidden_layers, args.epochs, args.pooler_type, args.seed)
                    os.system('rm ' + args.save_model_path + '/*.pth')
                    torch.save(student_model.bert.state_dict(),
                               args.save_model_path + '/' + now_save + '.pth'.format(steps))
                else:
                    logger.info(f"Step: {steps} / Best Results: {max_result} / Current Evaluation Results: {results}")
            steps += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Semantic Textual Similarity Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--data_path', type=str, default="./data/corpus/SS_Sym_VA_train_193M.txt", help='data_path')
    parser.add_argument('--num_hidden_layers', default=12, type=int, help='data_path')
    parser.add_argument('--save_model_path', type=str, default="./models/experiments", help='save_model_path')
    parser.add_argument('--student_model_path_or_name', type=str, default="vinai/phobert-base-v2",
                        help='student_model_path')
    parser.add_argument('--teacher_model_path_or_name', type=str,
                        default="./models/in_model/SS_Sym_VA_CTMethod_T1.2024", help='teacher_model_path')
    parser.add_argument('--eval_step', type=int, default=125, help='Eval Step')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='epochs')
    parser.add_argument('--num_workers', default=8, type=int, help='num_workers')
    parser.add_argument('--early_stop', default=3, type=int, help='early_stop')
    parser.add_argument('--loss_type', type=str, default="infonce",
                        choices=['mse', 'infonce'], help='Loss type')
    parser.add_argument('--pooler_type', type=str, default='cls', choices=['cls', 'cbp'], help='pooler_type')
    parser.add_argument('--use_vi_tokenizer', type=bool, default=True, help='use_vi_tokenizer')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    args = parser.parse_args()

    # set device
    torch.cuda.set_device(args.local_rank)
    main(args)
