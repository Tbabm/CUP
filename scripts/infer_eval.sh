#!/bin/bash

if [ $# -ne 4 ];then
  echo "wrong args"
  exit 1
fi

device=$1
dir=$2
model_class=$3
ds_dir=$4

model_path=${dir}/model.bin
test_set="${ds_dir}/test.jsonl"
output_file=${dir}/result.json

cd ..
CUDA_VISIBLE_DEVICES=${device} python infer.py --cuda \
                                               --model-class ${model_class} \
                                               ${model_path} \
                                               ${test_set} \
                                               ${output_file}

CUDA_VISIBLE_DEVICES=${device} python eval.py ${test_set} ${output_file}
