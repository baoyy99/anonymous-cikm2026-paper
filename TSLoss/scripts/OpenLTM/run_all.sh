#!/bin/bash
# nohup bash scripts/run_all.sh &

# 依次运行 4 个脚本
echo "==================== 开始运行第 1 个脚本 ===================="
bash scripts/OpenLTM/timer_xl_etth1.sh



echo "==================== 开始运行第 2 个脚本 ===================="
bash scripts/OpenLTM/other_etth1.sh

echo "==================== 所有脚本执行完毕！ ===================="