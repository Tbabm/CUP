#!/bin/bash

set -x -e -u -o pipefail

if [ -n "$1" ]; then
  device=$1
else
  device=0
fi

ds_dir="../../dataset"
log_dir="../logs"

echo "device: ${device}"
echo "dataset dir: ${ds_dir}"

ds_dir=$(realpath ${ds_dir})
cur_dir=$(pwd)

if [ ! -d ${log_dir} ];then
  mkdir ${log_dir}
fi

cd ..

test_a_model(){
  model_name=$1
  log_prefix=$2
  train_cmd=$3
  model_str=$4

  cd ${cur_dir}

  train_log="${log_dir}/${log_prefix}_train.log"
  infer_log="${log_dir}/${log_prefix}_infer.log"
  echo "train: ${model_str}" >> ${train_log}
  date >> ${train_log}
  ${train_cmd} ${device} ${log_prefix} models.updater.${model_name} ${ds_dir} >> ${train_log} 2>&1
  echo "test: ${model_str}" >> ${infer_log}
  date >> ${infer_log}
  ./infer_eval.sh ${device} ${log_prefix} models.updater.${model_name} ${ds_dir} >> ${infer_log}
}

#model_name="CoAttnBPBAUpdater"
#test_a_model ${model_name} ${model_name} "./train_model.sh" "${model_name}"

model_name="NoCoAttnBPBAUpdater"
test_a_model ${model_name} ${model_name} "./train_model.sh" "${model_name}"

model_name="CoAttnBPBAUpdater"
test_a_model ${model_name} ${model_name}_no_uni "./train_model_no_uni.sh" "${model_name} without unified vocabulary"

model_name="CoAttnBPBAUpdater"
test_a_model ${model_name} ${model_name}_no_ft "./train_model_no_ft.sh" "${model_name} without fastText"
