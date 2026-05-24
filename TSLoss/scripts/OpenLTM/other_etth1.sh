# model_name=gpt4ts
token_len=96
seq_len=96
pred_len=96
# a smaller batch size chosen due to large memory usage
# for GPT2 as backbone, can not use adaptation
# --learning_rate 1e-5 same as TSFM
for model in "gpt4ts" "time_llm" "timesfm"
# for model in "timesfm"
do
for loss in "mse" "mae" "cardloss" "timesql" "xpatchloss" "psloss" "fredfloss" "dbloss" "rankmseloss" "riloss" 
    # for loss in "timesql"
    do
python OpenLTM/run.py \
    --task_name forecast \
    --is_training 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id Ett \
    --model $model \
    --data MultivariateDatasetBenchmark  \
    --seq_len $seq_len \
    --input_token_len $token_len \
    --output_token_len $token_len \
    --test_seq_len $seq_len \
    --test_pred_len $pred_len \
    --d_model 768 \
    --d_ff 768 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --cosine \
    --train_epochs 10 \
    --use_norm \
    --pretrain_model_path OpenLTM/pretrained/gpt2 \
    --checkpoints OpenLTM/checkpoint\
    --gpu 0 \
    --gpt_layers 6 \
    --patch_size 16 \
    --stride 8 \
    --tmax 10 \
    --valid_last \
    --nonautoregressive \
    --subset_rand_ratio 0.05\
    --loss $loss
  done
done