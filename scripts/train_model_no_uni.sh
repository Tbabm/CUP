#!/bin/bash
# BEST!

if [ $# -ne 4 ];then
  echo "wrong args"
  exit 1
fi

device=$1
dir=$2
model_class=$3
ds_dir=$4
model_path="${dir}/model.bin"

echo "model_path ${model_path}"

cd ..
if [ ! -d $dir ];then
  mkdir $dir
fi
CUDA_VISIBLE_DEVICES=${device} python train.py \
                                       --train-data ${ds_dir}/train.jsonl \
                                       --dev-data ${ds_dir}/valid.jsonl \
                                       --vocab ${ds_dir}/vocab.json \
                                       --cuda \
                                       --input-feed \
                                       --share-embed \
                                       --dropout 0.2 \
                                       --use-pre-embed \
                                       --freeze-pre-embed \
                                       --vocab-embed ${ds_dir}/vocab_embeddings.pkl \
                                       --model-class ${model_class} \
                                       --log-dir ${dir} \
                                       --save-to ${model_path}
