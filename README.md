# CUP: JIT Comment UPdater
## Directory Structure
- baselines: the implementation of nnupdater and scripts for running baselines
- models: neural models
- scripts: the scripts for conducting experiments

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

## Download Dataset
- Our dataset, trained model and archived results can be downloaded from [here](https://drive.google.com/drive/folders/1WLkg1xvfxAwzFR6NWbEqZrTvr7QgQOkP?usp=sharing)
- Another archive of this project can be found at https://tinyurl.com/jitcomment
- By default, we store the dataset in `../dataset`

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
You can also build the vocabularies by yourself instead of using the one provided with our dataset.

```bash
# download fastText pre-trained model
cd ../dataset
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz

cd scripts
./build_vocab.sh ../../dataset/cc.en.300.bin
```

## Run Baselines
- Clone FracoUpdater:
```bash
# clone FracoUpdater
git clone https://github.com/Tbabm/FracoUpdater
```
- Install FracoUpdater's dependencies according to its [README](https://github.com/Tbabm/FracoUpdater/blob/master/README.md)
- Run
```
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


## Reference
If you use this code, please consider citing our paper

```
@inproceedings{liu2020automating,
  title={Automating Just-In-Time Comment Updating},
  author={Liu, Zhongxin and Xia, Xin and Yan, Meng and Li, Shanping},
  booktitle={Proceedings of the 35th IEEE/ACM International Conference on Automated Software Engineering},
  pages={713--725},
  year={2020}
}
```