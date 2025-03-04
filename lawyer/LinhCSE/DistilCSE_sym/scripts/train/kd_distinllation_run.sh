export CUDA_VISIBLE_DEVICES=0
num_hidden_layers=12
save_model_path=./models/newmse
data_dir=data/news5m.en
batch_size=512
lr=2e-4
epochs=20
num_workers=6
early_stop=3


python kd_distillation.py \
--data_dir ${data_dir} \
--num_hidden_layers $num_hidden_layers \
--eval_step 125 \
--save_model_path ${save_model_path} \
--batch_size ${batch_size} \
--lr ${lr} \
--epochs ${epochs} \
--num_workers ${num_workers} \
--early_stop ${early_stop}