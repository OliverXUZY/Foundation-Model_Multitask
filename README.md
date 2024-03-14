## Towards Few-Shot Adaptation of Foundation Models via Multitask Finetuning
Zhuoyan Xu, Zhenmei Shi, Junyi Wei, Fangzhou Mu, Yin Li, Yingyu Liang

This repository is the official Pytorch implementation of our method in the paper
[Towards Few-Shot Adaptation of Foundation Models via Multitask Finetuning](https://openreview.net/forum?id=1jbh2e0b2K) (ICLR 2024). 

In this repo, we provide code for multitask finetuning on various vision foundation models, along with proposed task selection algorithm. We provide the necessary code for multitask finetuning and ablation studies. 

## Requirements

It is tested under Ubuntu Linux 20.04 and Python 3.9 environment and requires some packages to be installed.

Pytorch >= 1.12.1 (guide is [here](https://pytorch.org/get-started/locally/))

Install other used packages:

```
pip install -r requirements.txt
```

**Datasets**
- [miniImageNet](https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE) (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))
- [tieredImageNet](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))
- [domainNet](http://ai.bu.edu/M3SDA/)
- [ImageNet-800](http://image-net.org/challenges/LSVRC/2012/)


### Usage

##### Get Started
- Modify the root (data, ckpt) in config file accordingly (`./configs`)

- To train and evaluate on mini-imagenet using clip ViT model, run

```bash
python finetune.py \
    --config=configs/clip/mini-imagenet/finetune_ViT.yaml \
    --n_batch_train 200 \
    --n_shot 1 \
    --sample_per_task 150
    
python test.py --config=configs/clip/mini-imagenet/test.yaml 
```

Please modify `path` in `test.yaml ` as saving model path or add `--path [path]` in command `python test.py`.


## Local data directory
```
datasets/
├── mini-imagenet
│   ├── miniImageNet_category_split_meta_train_limited100.pickle
│   ├── miniImageNet_category_split_meta_train_limited10.pickle
│   ├── miniImageNet_category_split_meta_train_limited20.pickle
│   ├── miniImageNet_category_split_meta_train_limited50.pickle
│   ├── miniImageNet_category_split_meta_train_limited_class16.pickle
│   ├── miniImageNet_category_split_meta_train_limited_class32.pickle
│   ├── miniImageNet_category_split_meta_train_limited_class8.pickle
│   ├── miniImageNet_category_split_test.pickle
│   ├── miniImageNet_category_split_train_phase_test.pickle
│   ├── miniImageNet_category_split_train_phase_train.pickle
│   ├── miniImageNet_category_split_train_phase_val.pickle
│   └── miniImageNet_category_split_val.pickle
```


### Citing

If you find our code useful, please consider citing:

```
@inproceedings{
    xu2024towards,
    title={Towards Few-Shot Adaptation of Foundation Models via Multitask Finetuning},
    author={Zhuoyan Xu and Zhenmei Shi and Junyi Wei and Fangzhou Mu and Yin Li and Yingyu Liang},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=1jbh2e0b2K}
}
```
