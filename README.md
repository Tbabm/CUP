# CUP: JIT Comment Updater
## Directory Structure
- baselines: the implementation of nnupdater and scripts for running baselines
- models: neural models
- scripts: the scripts for conductin experiments

## Prepare Requirements
- Java 8
- Install python dependencies through conda
- Install nlg-eval and setup

```bash
conda env create -f environment.yml

pip install git+https://github.com/Maluuba/nlg-eval.git@master
# set the data_path
nlg-eval --setup ${data_path}
```

## Train
```bash
cd scripts
# 0 for GPU 0
./train_model.sh 0 CoAttnBPBAUpdater models.updater.CoAttnBPBAUpdater ../dataset
```

## Infer and Evaluate
```bash
cd scripts
./infer_eval.sh 0 CoAttnBPBAUpdater models.updater.CoAttnBPBAUpdater ../dataset
```

## Build Vocab Yourself
```bash
# download fastText pre-trained model
cd ../dataset
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz

cd scripts
./build_vocab.sh ../../dataset/cc.en.300.bin
```

## Run Baselines
```bash
python -m baselines.run_baselines run_all_baselines
```

The results will be placed in the `dataset` directory, and can be evaluated using `CUP/eval.py`

## Run CUP's Variants
```bash
cd scripts
# 0 for GPU 0
./run_variants.sh 0
```

## Get Readable Result
```bash
python -m tools dump_all_readables
```

readable files can be found in results
