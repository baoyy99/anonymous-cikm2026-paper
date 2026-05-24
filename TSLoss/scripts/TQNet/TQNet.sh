export CUDA_VISIBLE_DEVICES=0
#!/bin/bash

seq_len=96
label_len=48
features="M"
embed="timeF"
target="OT"
freq="h"
train_epochs=10
patience=3

# ===================== Dataset =====================
# 索引0-8
dataset_name=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "traffic" "weather" "solar"  "electricity"  "exchange_rate")
root_path=("./dataset/ETT-small"
    "./dataset/ETT-small"
    "./dataset/ETT-small"
    "./dataset/ETT-small"
    "./dataset/traffic"
    "./dataset/weather"
    "./dataset/Solar"
    "./dataset/electricity"
    "./dataset/exchange_rate"
)
data_path=("ETTh1.csv"
    "ETTh2.csv"
    "ETTm1.csv"
    "ETTm2.csv"
    "traffic.csv"
    "weather.csv"
    "solar_AL.txt"
    "electricity.csv"
    "exchange_rate.csv"
)
data_type=("ETTh1"
    "ETTh2"
    "ETTm1"
    "ETTm2"
    "custom"
    "custom"
    "Solar"
    "custom"
    "custom"
)
enc_in=(7 7 7 7 862 21 137 321 8)
dec_in=(7 7 7 7 862 21 137 321 8)
c_out=(7 7 7 7 862 21 137 321 8)
batch_size=(32 32 32 32 16 32 32 32 32)
learning_rate=(0.0001 0.0001 0.0001 0.0001 0.001 0.0001 0.0001 0.0001 0.0001)
# ===================== Model =====================
lradj=('type3' 'type3' 'type3' 'type3' 'sigmoid' 'sigmoid' 'sigmoid' 'sigmoid' 'type1')
cycle=(24 24 96 96 168 144 144 168 24)
use_revin=(1 1 1 1 1 1 0 1 1)
# ==== begin: dataset → pred_len → loss  =====================
# 1、5轮随机种子，所有损失函数，预测长度96的情况
# 2、所有损失函数，预测长度其余3种的情况，4 5
# 3、新增两个损失函数，所有数据集预测长度96，5轮随机种子，
# 4、预测长度其余3种的etth1
# -一些杂七杂八的实验

for i in 0
# for i in {0..8}
do
#   for pred_len in 192 336 720
  for pred_len in 96 720
  do
    # for loss in "mse" "mae" "cardloss" "timesql" "xpatchloss" "fredfloss" "dbloss" "rankmseloss" "riloss" 
    for loss in "mae" "cardloss" "xpatchloss" "timesql" "dbloss"
    # for loss in "timesql"
    do
        # for loss_alpha in 0 1
        # do
        # for loss_beta in 0 1
        # do
        # for loss_gamma in 0.51 0.7 0.9
        # do
            # for random_seed in 2 20 200 2000 20000
            for random_seed in 2000
            do
                echo "----- ${dataset_name[$i]} ${pred_len} ${loss} ${random_seed} -----"
                python -u TQNet/run.py \
                    --is_training 0 \
                    --random_seed $random_seed \
                    --root_path ${root_path[$i]} \
                    --data_path ${data_path[$i]} \
                    --model TQNet \
                    --data ${data_type[$i]} \
                    --features $features \
                    --seq_len $seq_len \
                    --pred_len $pred_len \
                    --enc_in ${enc_in[$i]} \
                    --des 'Exp' \
                    --itr 1 \
                    --batch_size ${batch_size[$i]} \
                    --learning_rate ${learning_rate[$i]} \
                    --cycle ${cycle[$i]} \
                    --train_epochs ${train_epochs} \
                    --patience ${patience} \
                    --use_revin ${use_revin[$i]} \
                    --loss $loss\
                    --loss_alpha 0\
                    --loss_beta 0\
                    --loss_gamma 0
    done
done
done
done
# # done
# for i in 0
# do
#   for pred_len in 192 336 720
# #   for pred_len in 96
#   do
#     for loss in "mse" "mae" "softdtw" "cardloss" "timesql" "xpatchloss" "psloss" "fredfloss" "dbloss" "rankmseloss" "riloss" 
#     # for loss in "softdtw" "timesql"
#     do
#         # for loss_alpha in 0 1
#         # do
#         # for loss_beta in 0 1
#         # do
#         # for loss_gamma in 0.51 0.7 0.9
#         # do
#             # for random_seed in 2 20 200 2000 20000
#             # for random_seed in 2000
#             # do
#                 echo "----- ${dataset_name[$i]} ${pred_len} ${loss} ${random_seed} -----"
#                 python -u TQNet/run.py \
#                     --is_training 1 \
#                     --random_seed 2000 \
#                     --root_path ${root_path[$i]} \
#                     --data_path ${data_path[$i]} \
#                     --model TQNet \
#                     --data ${data_type[$i]} \
#                     --features $features \
#                     --seq_len $seq_len \
#                     --pred_len $pred_len \
#                     --enc_in ${enc_in[$i]} \
#                     --des 'Exp' \
#                     --itr 1 \
#                     --batch_size ${batch_size[$i]} \
#                     --learning_rate ${learning_rate[$i]} \
#                     --cycle ${cycle[$i]} \
#                     --train_epochs ${train_epochs} \
#                     --patience ${patience} \
#                     --use_revin ${use_revin[$i]} \
#                     --loss $loss\
#                     --loss_alpha 0\
#                     --loss_beta 0\
#                     --loss_gamma 0
#     done
# done
# done