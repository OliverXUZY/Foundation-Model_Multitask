import os
import pickle
import time
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import *
from torchvision.datasets import ImageFolder

# @register('meta-mixed-set')
# class mixedDataset(Dataset):
#     def __init__(self, root_mini, root_domain, root = None, split='meta-train', size=84, 
#                  n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, deterministic=False,
#                  n_view=1, n_meta_view=1, share_query=False, transform=None, val_transform=None):
#         """
#         Args:
#             All arguments needed for MetaMiniImageNet and MetaDomainNet.
#         """
#         root_mini = os.path.join("../datasets", "mini-imagenet")  # update for unsup-meta-mini-imagenet
#         self.dataset1 = MetaMiniImageNet(root=root_mini, split=split, size=size, 
#                                          n_view=n_view, n_meta_view=n_meta_view, share_query=share_query,
#                                          transform=transform, val_transform=val_transform,
#                                          n_batch=n_batch, n_episode=n_episode, n_way=n_way, n_shot=n_shot, n_query=n_query, deterministic=deterministic)

#         self.dataset2 = MetaDomainNet(root=root_domain, split=split, size=size, 
#                                       transform=transform, n_batch=n_batch, n_episode=n_episode, 
#                                       n_way=n_way, n_shot=n_shot, n_query=n_query, deterministic=deterministic)

#         self.length = min(len(self.dataset1), len(self.dataset2))

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         selected_dataset = np.random.choice([self.dataset1, self.dataset2])
#         item = selected_dataset[index]
#         return item
@register('meta-mixed-set')
class mixedDataset(Dataset):
    def __init__(self, root_mini, root_domain, root = None, split='meta-train', size=84, 
                 n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, deterministic=False,
                 n_view=1, n_meta_view=1, share_query=False, transform=None, val_transform=None):
        """
        Args:
            All arguments needed for MetaMiniImageNet and MetaDomainNet.
        """
        super(mixedDataset, self).__init__()

        

        self.statistics = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
        
        self.n_batch = n_batch
        self.n_episode = n_episode
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_shot_view = self.n_meta_view = n_meta_view
        self.deterministic = deterministic

        ##################### mini-imagenet ###########################
        split_dict = {'train': 'train_phase_train',        # standard train
                  'val': 'train_phase_val',            # standard val
                  'trainval': 'train_phase_trainval',  # standard train and val
                  'test': 'train_phase_test',          # standard test
                  'meta-train': 'train_phase_train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }
        split_tag = split_dict.get(split) or split
        split_file = '{}/miniImageNet_category_split_{}.pickle'.format(root_mini, split_tag)
        print("mini-imagenet: ", split_file)

        assert os.path.isfile(split_file)
        with open(split_file, 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
            data, label = pack['data'], pack['labels']

        data = [Image.fromarray(x) for x in data]
        label = np.array(label)
        label_key = sorted(np.unique(label))
        label_map = dict(zip(label_key, range(len(label_key))))
        new_label = np.array([label_map[x] for x in label])
        
        self.split_tag = split_tag
        self.size = size

        self.data = data
        self.label = new_label
        self.n_class = len(label_key)
        
        
        self.transform = MultiViewTransform(get_transform(transform, size, self.statistics), n_view)

        
        if share_query:
            self.n_query_view = 1
        else:
            self.n_query_view = n_meta_view

        self.catlocs = tuple()
        for cat in range(self.n_class):
            self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)

        self.val_transform = get_transform(val_transform, size, self.statistics)

        self.deterministic = deterministic

        ###################################################### domain-net ######################################################
        split_dict2 = {'train': 'train',        # standard train
                  'val': 'val',            # standard val
                  'test': 'test',          # standard test
                  'meta-train': 'train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }
        split_tag2 = split_dict2.get(split) or split
        split_dir2 = '{}/{}'.format(root_domain, split_tag2)
        print(split_dir2)
        assert os.path.isdir(split_dir2)
        
        self.transform2 = get_transform(transform, size, self.statistics)

        self.dataset2 = ImageFolder(root = split_dir2)

        if split_tag2 == "train":
            n_class2 = 180
        elif split_tag2 == "val":
            n_class2 = 65
        else:
            n_class2 = 100
        self.n_class2 = n_class2

        ### sampling part
        print("start sampling part dataset")
        ##### cache label file since it's time consuming
        cache_label_file = os.path.join(root_domain,"cached_{}_labels_domainNet.npy".format(split_tag2))
        if os.path.exists(cache_label_file):
            start = time.time()
            self.label = np.load(cache_label_file)
            print(
                f"Loading labels from cached file {cache_label_file} [took %.3f s]", time.time() - start
            )
        else:
            print(f"Creating labels from dataset file at {root}")
        
            start = time.time()
            self.label2 = np.array([target for _, target in self.dataset2])
            np.save(cache_label_file, self.label)
            # ^ This seems to take forever (but 5 mins at my laptop) so I want to investigate why and how we can improve.
            print(
                "Saving labels into cached file %s [took %.3f s]", cache_label_file, time.time() - start
            )

        self.catlocs2 = tuple()
        for cat in range(self.n_class2):
            self.catlocs2 += (np.argwhere(self.label == cat).reshape(-1),)

        print("construction dataset done!")
        

    def __len__(self):
        return self.n_batch * self.n_episode

    def _getitem_mini(self, index):
        if self.deterministic:
            np.random.seed(index)  ## add for control # of tasks and # of images
        s, q = self.n_shot, self.n_query
        sv, qv = self.n_shot_view, self.n_query_view
        shot, query = tuple(), tuple()
        
        cats = np.random.choice(self.n_class, self.n_way, replace=False)
        for c in cats:
            idx = np.random.choice(self.catlocs[c], sv * s + qv * q, replace=False)      # random choose n_shot*shot_view + n_query*query_view (1*1+15*1) images in each classes
            s_idx, q_idx = idx[:sv * s], idx[-qv * q:]
            c_shot = torch.stack([self.transform(self.data[i]) for i in s_idx])          # [1,1,3,84,84] [S*SV, V, C, H ,W]
            c_query = torch.stack([self.val_transform(self.data[i]) for i in q_idx])     # [15,3,84,84] [Q*QV, C, H ,W]
            c_shot = c_shot.view(sv, s, *c_shot.shape[-4:])                              # [1,1,1,3,84,84] [SV, S, V, C, H ,W]
            c_query = c_query.view(qv, q, *c_query.shape[-3:])                           # [1,10,3,84,84] [QV, Q, C, H ,W]
            shot += (c_shot,)
            query += (c_query,)
        
        shot = torch.cat(shot, dim=1)      # [SV, Y * S, V, C, H, W]                   # [1, 5, 1, 3, 84, 84] 
        query = torch.cat(query, dim=1)    # [QV, Y * Q, C, H, W]                      # [1, 50, 3, 84, 84]
        cats = torch.from_numpy(cats)
        return shot, query, cats
    
    def _getitem_domain(self, index):
        if self.deterministic:
            np.random.seed(index)  ## add for control # of tasks and # of images
        s, q = self.n_shot, self.n_query
        sv, qv = 1, 1
        shot, query = tuple(), tuple()
        
        cats = np.random.choice(self.n_class2, self.n_way, replace=False)
        for c in cats:
            idx = np.random.choice(self.catlocs2[c], sv * s + qv * q, replace=False)      # random choose n_shot*shot_view + n_query*query_view (1*1+15*1) images in each classes
            s_idx, q_idx = idx[:sv * s], idx[-qv * q:]
            # print([self.transform(self.dataset[i][0]).shape for i in s_idx])
            # print([self.transform(self.dataset[i][0]).shape for i in q_idx])
            c_shot = torch.stack([self.transform2(self.dataset2[i][0]) for i in s_idx])          # [5(1),3,84,84] [S*SV, C, H ,W]
            c_query = torch.stack([self.transform2(self.dataset2[i][0]) for i in q_idx])         # [15,3,84,84] [Q*QV, C, H ,W]
            c_shot = c_shot.view(sv, s, 1, *c_shot.shape[-3:])   # hard code V = 1             # [1,5(1),1,3,84,84] [SV, S, V, C, H ,W]
            c_query = c_query.view(qv, q, *c_query.shape[-3:])                           # [1,15,3,84,84] [QV, Q, C, H ,W]
            shot += (c_shot,)
            query += (c_query,)
        
        shot = torch.cat(shot, dim=1)      # [SV, Y * S, V, C, H, W]                   # [1, 5, 1, 3, 84, 84] 
        query = torch.cat(query, dim=1)    # [QV, Y * Q, C, H, W]                      # [1, 50, 3, 84, 84]
        cats = torch.from_numpy(cats)
        return shot, query, cats
    
    def __getitem__(self, index):
        selected_dataset = np.random.choice([self._getitem_mini, self._getitem_domain])
        # print("select getitem method", selected_dataset)
        item = selected_dataset(index)
        return item