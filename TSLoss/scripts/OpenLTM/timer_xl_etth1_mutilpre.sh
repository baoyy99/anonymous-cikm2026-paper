export CUDA_VISIBLE_DEVICES=0
model_name=timer_xl
token_num=30
token_len=96
seq_len=$[$token_num*$token_len]
for loss in "mse" "mae" "softdtw" "cardloss" "timesql" "xpatchloss" "psloss" "fredfloss" "dbloss" "rankmseloss" "riloss" 
    # for loss in "mse"
do

for test_pred_len in 192 336 720 96
do
  python -u OpenLTM/run.py\
    --task_name forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/\
    --data_path ETTh1.csv \
    --model_id ETTh1_few_shot \
    --model $model_name \
    --data MultivariateDatasetBenchmark  \
    --seq_len $seq_len \
    --input_token_len $token_len \
    --output_token_len $token_len \
    --test_seq_len $seq_len \
    --test_pred_len $test_pred_len \
    --e_layers 8 \
    --d_model 1024 \
    --d_ff 2048 \
    --batch_size 16 \
    --learning_rate 5e-6 \
    --train_epochs 10 \
    --gpu 0 \
    --cosine \
    --tmax 10 \
    --use_norm \
    --adaptation \
    --pretrain_model_path OpenLTM/pretrained/timer_xl_260b/checkpoint.pth \
    --checkpoints OpenLTM/checkpoint\
    --subset_rand_ratio 0.05\
    --loss $loss
  done
done