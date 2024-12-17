import sys

sys.path.append('..')
import json
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ram import get_transform


class VOC2012_handler(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = 'D:\datasets\mll_datasets/voc/VOCdevkit/VOC2012'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/JPEGImages/' + self.X[index]).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class COCO2014_handler(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = 'D:\datasets\mll_datasets/coco'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class CUB_200_2011_handler(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = 'D:\datasets\mll_datasets/cub/CUB_200_2011/images'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class NUS_WIDE_handler(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = 'D:\datasets\mll_datasets/nus/Flickr'

    def __getitem__(self, index):
        x = Image.open(self.data_path+'/'+self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


def load_spml_datasets(
        dataset: str,
        model_type: str,
        pattern: str,
        input_size: int,
        batch_size: int,
        num_workers: int
) -> Tuple[DataLoader, Dict]:
    dataset_root = "D:\datasets\mll_datasets/" + dataset
    # Label system of tag2text contains duplicate tag texts, like
    # "train" (noun) and "train" (verb). Therefore, for tag2text, we use
    # `tagid` instead of `tag`.
    if model_type == "ram_plus" or model_type == "ram":
        tag_file = dataset_root + f"/{dataset}_ram_taglist.txt"
    else:
        tag_file = dataset_root + f"/{dataset}_tag2text_tagidlist.txt"

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    imglist = np.load(os.path.join(f'D:\datasets\mll_datasets/{dataset}', f'formatted_{pattern}_images.npy'))

    if dataset == "voc":
        train_dataset = VOC2012_handler(X=np.load(os.path.join('D:\datasets\mll_datasets/voc', 'formatted_train_images.npy')),
                                        Y=np.load(os.path.join('D:\datasets\mll_datasets/voc', 'formatted_train_labels.npy')),
                                        input_size=input_size)
        test_dataset = VOC2012_handler(X=np.load(os.path.join('D:\datasets\mll_datasets/voc', 'formatted_val_images.npy')),
                                       Y=np.load(os.path.join('D:\datasets\mll_datasets/voc', 'formatted_val_labels.npy')),
                                       input_size=input_size)
    elif dataset == "coco":
        train_dataset = COCO2014_handler(X=np.load(os.path.join('D:\datasets\mll_datasets/coco', 'formatted_train_images.npy')),
                                         Y=np.load(os.path.join('D:\datasets\mll_datasets/coco', 'formatted_train_labels.npy')),
                                         input_size=input_size)
        test_dataset = COCO2014_handler(X=np.load(os.path.join('D:\datasets\mll_datasets/coco', 'formatted_val_images.npy')),
                                        Y=np.load(os.path.join('D:\datasets\mll_datasets/coco', 'formatted_val_labels.npy')),
                                        input_size=input_size)
    elif dataset == "cub":
        train_dataset = CUB_200_2011_handler(X=np.load(os.path.join('D:\datasets\mll_datasets/cub', 'formatted_train_images.npy')),
                                             Y=np.load(
                                                 os.path.join('D:\datasets\mll_datasets/cub', 'formatted_train_labels_obs.npy')),
                                             input_size=input_size)
        test_dataset = CUB_200_2011_handler(X=np.load(os.path.join('D:\datasets\mll_datasets/cub', 'formatted_val_images.npy')),
                                            Y=np.load(os.path.join('D:\datasets\mll_datasets/cub', 'formatted_val_labels.npy')),
                                            input_size=input_size)
    elif dataset == "nus":
        train_dataset = NUS_WIDE_handler(X=np.load(os.path.join('D:\datasets\mll_datasets/nus', 'formatted_train_images.npy')),
                                         Y=np.load(os.path.join('D:\datasets\mll_datasets/nus', 'formatted_train_labels_obs.npy')),
                                         input_size=input_size)
        test_dataset = NUS_WIDE_handler(X=np.load(os.path.join('D:\datasets\mll_datasets/nus', 'formatted_val_images.npy')),
                                        Y=np.load(os.path.join('D:\datasets\mll_datasets/nus', 'formatted_val_labels.npy')),
                                        input_size=input_size)

    if pattern == "train":
        loader = DataLoader(dataset=train_dataset, shuffle=True, drop_last=False, pin_memory=True,
                            batch_size=batch_size, num_workers=num_workers)
    if pattern == "val":
        loader = DataLoader(dataset=test_dataset, shuffle=False, drop_last=False, pin_memory=True,
                            batch_size=batch_size, num_workers=num_workers)

    open_tag_des = dataset_root + f"/{dataset}_llm_tag_descriptions.json"
    if os.path.exists(open_tag_des):
        with open(open_tag_des, 'rb') as fo:
            tag_des = json.load(fo)

    else:
        tag_des = None
    info = {
        "taglist": taglist,
        "imglist": imglist,
        # "annot_file": annot_file,
        # "img_root": img_root,
        "tag_des": tag_des
    }

    return loader, info


def get_split_datasets(dataset, model_type, pattern, input_size, batch_size, num_workers):
    dataset_root = "../datasets/" + dataset
    # Label system of tag2text contains duplicate tag texts, like
    # "train" (noun) and "train" (verb). Therefore, for tag2text, we use
    # `tagid` instead of `tag`.
    if model_type == "ram_plus" or model_type == "ram":
        tag_file = dataset_root + f"/{dataset}_ram_taglist.txt"
    else:
        tag_file = dataset_root + f"/{dataset}_tag2text_tagidlist.txt"

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    imglist = np.load(os.path.join(f'../datasets/{dataset}', f'formatted_{pattern}_images.npy'))

    if dataset == "voc":
        train_dataset = VOC2012_handler(X=np.load(os.path.join('../datasets/voc', 'formatted_train_images.npy')),
                                        Y=np.load(os.path.join('../datasets/voc', 'formatted_train_labels_obs.npy')),
                                        input_size=input_size)
        test_dataset = VOC2012_handler(X=np.load(os.path.join('../datasets/voc', 'formatted_val_images.npy')),
                                       Y=np.load(os.path.join('../datasets/voc', 'formatted_val_labels.npy')),
                                       input_size=input_size)
    elif dataset == "coco":
        train_dataset = COCO2014_handler(X=np.load(os.path.join('../datasets/coco', 'formatted_train_images.npy')),
                                         Y=np.load(os.path.join('../datasets/coco', 'formatted_train_labels_obs.npy')),
                                         data_path='../datasets/coco',
                                         input_size=input_size)
        test_dataset = COCO2014_handler(X=np.load(os.path.join('../datasets/coco', 'formatted_val_images.npy')),
                                        Y=np.load(os.path.join('../datasets/coco', 'formatted_val_labels.npy')),
                                        data_path='../datasets/coco',
                                        input_size=input_size)
    elif dataset == "cub":
        train_dataset = CUB_200_2011_handler(X=np.load(os.path.join('../datasets/cub', 'formatted_train_images.npy')),
                                             Y=np.load(
                                                 os.path.join('../datasets/cub', 'formatted_train_labels_obs.npy')),
                                             input_size=input_size)
        test_dataset = CUB_200_2011_handler(X=np.load(os.path.join('../datasets/cub', 'formatted_val_images.npy')),
                                            Y=np.load(os.path.join('../datasets/cub', 'formatted_val_labels.npy')),
                                            input_size=input_size)

    labels = train_dataset.Y.astype(int)
    positive_index = []
    negative_index = []
    for i in range(len(labels)):
        if labels[i].sum() == 1:
            positive_index.append(i)
        else:
            negative_index.append(i)
    # positive_index = torch.tensor(positive_index)
    # negative_index = torch.tensor(negative_index)
    # # torch.index_select(labels, dim=0, index=positive_index)
    if dataset == "voc":
        train_single_pos_dataset = VOC2012_handler(X=train_dataset.X[positive_index],
                                                   Y=train_dataset.Y[positive_index],
                                                   input_size=input_size)
        train_single_neg_datatset = VOC2012_handler(X=train_dataset.X[negative_index],
                                                    Y=train_dataset.Y[negative_index],
                                                    input_size=input_size)
    elif dataset == "coco":
        train_single_pos_dataset = COCO2014_handler(X=train_dataset.X[positive_index],
                                                    Y=train_dataset.Y[positive_index],
                                                    data_path='../datasets/coco',
                                                    input_size=input_size)
        train_single_neg_datatset = COCO2014_handler(X=train_dataset.X[negative_index],
                                                     Y=train_dataset.Y[negative_index],
                                                     data_path='../datasets/coco',
                                                     input_size=input_size)
    elif dataset == "cub":
        train_single_pos_dataset = CUB_200_2011_handler(X=train_dataset.X[positive_index],
                                                        Y=train_dataset.Y[positive_index],
                                                        input_size=input_size)
        train_single_neg_datatset = CUB_200_2011_handler(X=train_dataset.X[negative_index],
                                                         Y=train_dataset.Y[negative_index],
                                                         input_size=input_size)

    train_pos_loader = DataLoader(dataset=train_single_pos_dataset, shuffle=True, drop_last=False, pin_memory=True,
                                  batch_size=batch_size, num_workers=num_workers)
    train_neg_loader = DataLoader(dataset=train_single_neg_datatset, shuffle=True, drop_last=False, pin_memory=True,
                                  batch_size=batch_size, num_workers=num_workers)

    test_loader = DataLoader(dataset=test_dataset, shuffle=False, drop_last=False, pin_memory=True,
                             batch_size=batch_size, num_workers=num_workers)

    open_tag_des = dataset_root + f"/{dataset}_llm_tag_descriptions.json"
    if os.path.exists(open_tag_des):
        with open(open_tag_des, 'rb') as fo:
            tag_des = json.load(fo)

    else:
        tag_des = None
    info = {
        "taglist": taglist,
        "imglist": imglist,
        # "annot_file": annot_file,
        # "img_root": img_root,
        "tag_des": tag_des
    }

    return train_pos_loader, train_neg_loader, test_loader, info
