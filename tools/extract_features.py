import argparse
import os
import json
import gc
import random
import sys
sys.path.append("/srv/home/zxu444/few_shot_image")
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
from torchvision.datasets import ImageFolder

import datasets
import models
from models import encoders, classifiers
import utils
import utils.optimizers as optimizers

import clip

templates = ['a photo of a {}',
             'itap of a {}.',
            'a bad photo of the {}.',
            'a origami {}.',
            'a photo of the large {}.',
            'a {} in a video game.',
            'art of the {}.',
            'a photo of the small {}.']


data_root = "/srv/home/zxu444/datasets"

def encode_name_to_feature(name, clip_model):
    '''
    name: str: 'electric guitar'

    clip_model: model.encode_text in CLIP model (enc.model)
    '''
    with torch.no_grad():
        text_embedding_templates = list(map(
            lambda template: clip_model.encode_text(clip.tokenize(
            template.format(name)                  # [1,embdeeing_size] [1,77]
            ).cuda(non_blocking=True)),            # [1,D]   [1,512]
            templates
        ))
    s = torch.concat(text_embedding_templates)       # [8, D] [8, 512]
    s /= s.norm(dim = -1, keepdim=True)     # normalize
    # print(s.dtype)
    s = torch.mean(s, dim=0)                         # [1, D] [1, 512]
    s /= s.norm(dim = -1, keepdim=True)
    # print(s.shape, s.dtype, s.norm()
    return s



def text_encoder(shot_names, clip_model):
    '''
    shot_names: list with length Y, each element of list is a tuple with E names
    # [('vase', 'Alaskan Malamute', 'electric guitar', 'mixing bowl'),
    # ('red king crab', 'scoreboard', 'Dalmatian', 'Golden Retriever'),
    # ('trifle', 'lion', 'vase', 'red king crab'),
    # ('black-footed ferret', 'crate', 'nematode', 'front curtain'),
    # ('crate', 'Golden Retriever', 'bookstore', 'Dalmatian')]

    clip_model: model.encode_text in CLIP model (enc.model)
    '''
    s = []
    for template in templates:
        token_ep = list(map(
            lambda x: clip_model.encode_text(     # x = ('vase', 'Alaskan Malamute', 'electric guitar', 'mixing bowl')
                torch.concat(
                    [clip.tokenize(template.format(name)) for name in x]   # each element is [1,77] tokens for one sentence
                    ).cuda(non_blocking=True)                               # 4 eps ('vase', 'Alaskan Malamute', 'electric guitar', 'mixing bowl') for tokens for one of Y class [E,77] [4, 77]
                ),        # 4 eps for text features for one of Y class  [E,D] [4, 512]
            shot_names
            ))            # list with length Y, each element is above [E,D]

        textfea_ep = torch.stack(token_ep)      # [Y,E,D] [5, 4, 512]
        s.append(textfea_ep)
    s = torch.stack(s)                      # [T=8,Y,E,D] 8 templates        [8, 5, 4, 512]
    s = s.transpose(0,2)                    # [E,Y,T,D]      [4, 5, 8, 512]
    s /= s.norm(dim = -1, keepdim=True)     # normalize

    return s

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a vision encoder on vision language model")
    parser.add_argument(
        '--data_root', help='data dir'
    )
    parser.add_argument(
        "--model", help="model name"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    enc = encoders.make("ViT-B32")
    # encode_name_to_feature('vase', enc.model)

    with open(os.path.join(data_root,'classnames.txt')) as f:
        lines = [line.rstrip() for line in f]

    class_to_name = {}
    for line in lines:
        s_id = line.find(' ')
        class_to_name[line[:s_id]] = line[s_id+1:]
    
    root_dir = '/srv/home/zxu444/datasets/mini-imagenet'


    # for split_tag in ['val', 'test', 'train']:
    split_tag = "test"
    print("start mini_imagenet {} data".format(split_tag))
    start = time.time()
    name_to_textRep = {}
    split_dir = '{}/{}'.format(root_dir, split_tag)
    dataset = ImageFolder(root = split_dir)
    idx_to_name = {}
    for c in dataset.class_to_idx:
        idx_to_name[dataset.class_to_idx[c]] = class_to_name[c]
    for idx, name in idx_to_name.items():
        # print(idx, name)
        # print("memory_allocated: ", torch.cuda.memory_allocated())
        # print("memory_reserved: ", torch.cuda.memory_reserved())
        fea = encode_name_to_feature(name, enc.model)
        name_to_textRep[name] = fea.cpu().data.numpy().tolist()
        del fea
        torch.cuda.empty_cache()
    
    
    print("Done mini_imagenet {} data. [took {:.3f} s]".format(split_tag, time.time() - start))
    # for name, textFea in name_to_textRep.items():
    #     print("{}: {}".format(name, textFea.shape))
    
    # del name_to_textRep
    # n = gc.collect()
    # torch.cuda.empty_cache()
    
    j = json.dumps(name_to_textRep)
    with open(os.path.join(root_dir, "tiered-imagenet_{}_ViT-B32_text_representation.json").format(split_tag),"w") as f:
        f.write(j)
    
    
    
   


if __name__ == "__main__":
    main()