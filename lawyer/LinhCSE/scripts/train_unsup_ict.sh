export PYTHONPATH=./
timestamp=`date "+%Y%0m%0d_%T"`
# datas
data_dir="/home/link/spaces/chunking/LinhCSE_training/data"
benchmark_dir="/home/link/spaces/chunking/LinhCSE_training/benchmark_concat"
model_dir="models/ckpt_$timestamp"
checkpoint="vinai/phobert-base-v2"

# logs
wandb_run_name="example-run-name"

# params
s="123"
lr="5e-5"

CUDA_VISIBLE_DEVICES=0 python3 cse/runs/main.py \
        --model_dir $model_dir \
        --data_dir $data_dir \
        --benchmark_dir $benchmark_dir \
        --token_level '' \
        --model_type kid-dense-sim-cse-vietnamese \
        --logging_steps 5 \
        --save_steps 5 \
        --wandb_run_name $wandb_run_name \
        --do_eval \
        --seed $s \
        --num_train_epochs 10 \
        --train_batch_size 768 \
        --eval_batch_size 512 \
        --max_seq_len_query 64 \
        --max_seq_len_document 256 \
        --learning_rate $lr \
        --tuning_metric recall_bm_newlegal_a_5 \
        --early_stopping 5 \
        --resize_embedding_model \
        --pooler_type avg \
        --gradient_checkpointing \
        --sim_fn dot \
        --use_lowercase True \
        --use_remove_punc True \
        --pretrained \
        --pretrained_path $checkpoint \
        --do_train
