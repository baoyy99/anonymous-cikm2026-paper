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

e_layers=(2 2 2 2 4 3 3 3 2 )
d_model=(128 128 128 128 512 512 512 512 128)
d_ff=(128 128 128 128 512 512 512 512 128)
# # # ==== begin: dataset → pred_len → loss =====================
for i in 4
# for i in 0
do
#   for pred_len in 96 192 336 720
  for pred_len in 96
  do
    # for loss in "mse" "mae" "cardloss" "timesql" "xpatchloss" "psloss" "fredfloss" "dbloss" "rankmseloss" "riloss" 
    # for loss in "mse" "mae" "cardloss" "xpatchloss" "psloss" "fredfloss" "dbloss" "rankmseloss" "riloss" 
    for loss in "riloss"
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
                    --is_training 1 \
                    --random_seed $random_seed \
                    --root_path ${root_path[$i]} \
                    --data_path ${data_path[$i]} \
                    --model iTransformer \
                    --data ${data_type[$i]} \
                    --features $features \
                    --seq_len $seq_len \
                    --pred_len $pred_len \
                    --e_layers ${e_layers[$i]} \
                    --d_layers 1 \
                    --factor 3 \
                    --enc_in ${enc_in[$i]} \
                    --dec_in ${dec_in[$i]} \
                    --c_out ${c_out[$i]} \
                    --d_model ${d_model[$i]} \
                    --d_ff ${d_ff[$i]} \
                    --des 'Exp' \
                    --itr 1 \
                    --batch_size ${batch_size[$i]} \
                    --learning_rate ${learning_rate[$i]} \
                    --train_epochs ${train_epochs} \
                    --patience ${patience} \
                    --loss $loss\
                    --loss_alpha 0\
                    --loss_beta 0\
                    --loss_gamma 0
    done
done  
done
done


# # ====================偶尔使用，只测试======================
# # for i in {0..8}
# # for i in 7
# for i in 8
# do
#   for pred_len in 96
#   do
#     # echo "========== 当前预测长度：$pred_len =========="
#         echo "----- ${dataset_name[$i]} ${pred_len} -----"
#         python -u 12-tslib-asymmetry/run.py \
#             --is_training 0 \
#             --task_name long_term_forecast \
#             --model_id $pred_len\
#             --root_path ${root_path[$i]} \
#             --data_path ${data_path[$i]} \
#             --model iTransformer \
#             --data ${data_type[$i]} \
#             --features $features \
#             --seq_len $seq_len \
#             --pred_len $pred_len \
#             --e_layers ${e_layers[$i]} \
#             --d_layers 1 \
#             --factor 3 \
#             --enc_in ${enc_in[$i]} \
#             --dec_in ${dec_in[$i]} \
#             --c_out ${c_out[$i]} \
#             --d_model ${d_model[$i]} \
#             --d_ff ${d_ff[$i]} \
#             --des 'Exp' \
#             --itr 1 \
#             --batch_size ${batch_size[$i]} \
#             --learning_rate ${learning_rate[$i]} \
#             --train_epochs ${train_epochs} \
#             --patience ${patience} \
#             --loss $loss\
#             --loss_alpha 0\
#             --loss_beta 1\
#             --loss_gamma 0.1\
#             --loss_q 0.52
#     done
# done


# # ====================四种情况======================
# for i in {0..8}
# for i in 0
# do
#   for pred_len in 96
#   do
#     for loss_gamma in 0.1 0.2 0.3
#     do
#         for loss_alpha in 1 0 
#         do
#             for loss_beta in 1 0 
#             do
#                 # echo "========== 当前预测长度：$pred_len =========="
#                     # echo "----- ${dataset_name[$i]} ${pred_len} -----"
#                     python -u 12-tslib-asymmetry/run.py \
#                         --is_training 1 \
#                         --task_name long_term_forecast \
#                         --model_id $pred_len\
#                         --root_path ${root_path[$i]} \
#                         --data_path ${data_path[$i]} \
#                         --model iTransformer \
#                         --data ${data_type[$i]} \
#                         --features $features \
#                         --seq_len $seq_len \
#                         --pred_len $pred_len \
#                         --e_layers ${e_layers[$i]} \
#                         --d_layers 1 \
#                         --factor 3 \
#                         --enc_in ${enc_in[$i]} \
#                         --dec_in ${dec_in[$i]} \
#                         --c_out ${c_out[$i]} \
#                         --d_model ${d_model[$i]} \
#                         --d_ff ${d_ff[$i]} \
#                         --des 'Exp' \
#                         --itr 1 \
#                         --batch_size ${batch_size[$i]} \
#                         --learning_rate ${learning_rate[$i]} \
#                         --train_epochs ${train_epochs} \
#                         --patience ${patience} \
#                         --loss $loss\
#                         --loss_alpha $loss_alpha\
#                         --loss_beta $loss_beta\
#                         --loss_gamma $loss_gamma\
#                         --loss_q 0.52
#                 done
#             done
#         done
#     done
# done