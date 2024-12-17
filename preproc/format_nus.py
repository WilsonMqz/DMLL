import json
import os
import argparse
import numpy as np
import torch

pp = argparse.ArgumentParser(description='Format NUS metadata.')
pp.add_argument('--load-path', type=str, default='../datasets/nus', help='Path to a directory containing a copy of the NUS dataset.')
pp.add_argument('--save-path', type=str, default='../datasets/nus', help='Path to output directory.')
args = pp.parse_args()

train_imagelist = []
val_imagelist = []

with open(os.path.join(args.load_path, "TrainImagelist.txt"), 'r') as file:
    for line in file:
        content = line.strip()  # 去除字符串两端的空白字符

        # 使用相应的字符串操作提取部分内容
        parts = content.split('\\')

        # 提取各个部分
        drive = parts[0]
        folder1 = parts[1]
        folder2 = parts[2]
        folder3 = parts[3]
        filename = parts[4]

        image_loc = folder3+"/"+filename
        train_imagelist.append(image_loc)
with open(os.path.join(args.load_path, "TestImagelist.txt"), 'r') as file:
    for line in file:
        content = line.strip()  # 去除字符串两端的空白字符

        # 使用相应的字符串操作提取部分内容
        parts = content.split('\\')

        # 提取各个部分
        drive = parts[0]
        folder1 = parts[1]
        folder2 = parts[2]
        folder3 = parts[3]
        filename = parts[4]

        image_loc = folder3+"/"+filename
        val_imagelist.append(image_loc)

train_imagelist = np.array(train_imagelist)
val_imagelist = np.array(val_imagelist)

np.save(os.path.join(args.save_path, 'formatted_' + 'train' + '_images.npy'), train_imagelist)
np.save(os.path.join(args.save_path, 'formatted_' + 'val' + '_images.npy'), val_imagelist)

num_train = len(np.load(os.path.join(args.load_path, "formatted_train_images.npy")))
num_val = len(np.load(os.path.join(args.load_path, "formatted_val_images.npy")))
assert num_train==161789
assert num_val==107859

train_label_matrix = np.zeros((num_train,81),dtype=float)
val_label_matrix = np.zeros((num_val,81),dtype=float)
cls_idx = 0

with open(os.path.join(args.load_path, "Concepts81.txt"), 'r') as file1:
    for o_line1 in file1:
        line1 = o_line1.rstrip('\n')
        train_label_path = args.load_path + "/TrainTestLabels" + "/" + "Labels_" + line1 + "_Train.txt"
        val_label_path = args.load_path + "/TrainTestLabels" + "/" + "Labels_" + line1 + "_Test.txt"
        train_idx = 0
        val_idx = 0
        with open(train_label_path,'r') as file2:
            for o_line2 in file2:
                line2 = o_line2.rstrip('\n')
                trlab_value = float(line2)
                train_label_matrix[train_idx,cls_idx] = trlab_value
                train_idx += 1
        with open(val_label_path,'r') as file3:
            for o_line3 in file3:
                line3 = o_line3.rstrip('\n')
                vallab_value = float(line3)
                val_label_matrix[val_idx,cls_idx] = vallab_value
                val_idx += 1
        cls_idx += 1
        
np.save(os.path.join(args.save_path, 'formatted_' + 'train' + '_labels.npy'), train_label_matrix)
np.save(os.path.join(args.save_path, 'formatted_' + 'val' + '_labels.npy'), val_label_matrix)

    
                



