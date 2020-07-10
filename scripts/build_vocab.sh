#!/bin/bash

if [ $# -ne 1 ];then
  echo "Please provide the path to the fastText model. E.g. ../dataset/cc.en.300.bin"
  exit 1
fi

ft_model=$(realpath $1)

cd ..
ds_dir="../dataset"

build_vocab(){
    em_file=$1
    vocab_class=$2
    vocab_file=$3

    python vocab.py --train-set ${ds_dir}/train.jsonl --use-ft \
                  --ft-model ${ft_model} \
                  --embedding-file ${ds_dir}/${em_file} \
                  --vocab-class ${vocab_class} \
                  ${ds_dir}/${vocab_file}
}

# build normal vocab
build_vocab "vocab_embeddings.pkl" "Vocab" "vocab.json"

# build unified vocab
build_vocab "mix_vocab_embeddings.pkl" "MixVocab" "mix_vocab.json"
