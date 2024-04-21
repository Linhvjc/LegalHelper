local_rank=0
seed=1111
data_path=/home/link/spaces/asym_data/train/data.jsonl
save_model_path=./models/experiments/ckd
# student_model_path_or_name=./models/in_model/SS_Sym_VA_CTMethod_T1.2024
student_model_path_or_name=vinai/phobert-base-v2
teacher_model_path_or_name=BAAI/bge-m3

CUDA_VISIBLE_DEVICES=0 python trainer/ckd_contrastive.py \
--local_rank ${local_rank} \
--batch_size 128 \
--data_path ${data_path} \
--max_seq_len 128 \
--save_model_path ${save_model_path} \
--student_model_path_or_name ${student_model_path_or_name} \
--teacher_model_path_or_name ${teacher_model_path_or_name} \
--eval_steps 300 \
--queue_len 50000 \
--epochs 10 \
--num_workers 16 \
--lr 5e-5 \
--temp 1 \
--temp_exp 2 \
--early_stop 5 \
--pooler_type cls \
--use_vi_tokenizer true \
--seed ${seed} \
--mse 0
