export CUDA_VISIBLE_DEVICES=0
local_rank=0
seed=1111
model_base=vinai/phobert-base-v2
model_path=./models/experiments/ckd/eval_step_375-epoch_10-pooler_cls-seed_1111.pth
data_path=./data/labeled/VI_NLI/processed/UIT_ViNLI.json
save_model_path=./models/experiments/nli

python trainer/nli_cls_finetune.py \
--local_rank ${local_rank} \
--data_path ${data_path} \
--model_base ${model_base} \
--model_path ${model_path} \
--lr 2e-5 \
--batch_size 128 \
--save_model_path ${save_model_path} \
--epochs 10 \
--eval_step 125 \
--max_seq_len 128 \
--pooler_type cls \
--loss_type cross_entropy \
--poly_reduce_type mean \
--use_vi_tokenizer true \
--seed ${seed}
