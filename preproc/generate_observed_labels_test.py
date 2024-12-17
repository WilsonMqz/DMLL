import numpy as np
import os
import argparse

import torch

pp = argparse.ArgumentParser(description='')
pp.add_argument('--dataset', type=str, choices=['voc', 'coco', 'nus', 'cub'], default='cub')
pp.add_argument('--num-labeled', type=int, default=10, help='need labeled num')
pp.add_argument('--seed', type=int, default=1200, help='random seed')
args = pp.parse_args()

np.random.seed(args.seed)


def get_random_label_indices(label_vec, label_value, num_sel, rng):
    '''
    Given a 1D numpy array label_vec, return num_sel indices chosen at random
    from all indices where label_vec equals label_value.
    Note that all relevant indices are returned if fewer than num_sel are found.
    '''

    # find all indices where label_vec is equal to label_value:
    idx_all = np.nonzero(label_vec == label_value)[0]

    # shuffle those indices:
    rng.shuffle(idx_all)

    # return (up to) the first num_sel indices:
    idx_sel = idx_all[:num_sel]

    return idx_sel


def observe_uniform(label_matrix, num_labeled):
    '''
    label_matrix: binary (-1/+1) label matrix with shape num_items x num_classes
    num_pos: number of positive labels to observe for each item
    num_neg: number of negative labes to observe for each item
    rng: random number generator to use
    '''

    # check that label_matrix is a binary numpy array:
    assert type(label_matrix) is np.ndarray
    label_values = np.unique(label_matrix)
    assert len(label_values) == 2
    assert 1 in label_values

    # apply uniform observation process:
    num_items, num_classes = np.shape(label_matrix)
    label_matrix_obs = np.zeros_like(label_matrix)

    labeled_index = np.random.choice(np.arange(num_items), size=num_labeled, replace=False)
    for i in range(len(labeled_index)):
        label_matrix_i = label_matrix[labeled_index[i]]
        label_matrix_i[label_matrix_i == 0] = -1
        label_matrix_obs[labeled_index[i]] = label_matrix[labeled_index[i]]

    print(label_matrix_obs[labeled_index])

    positive_class = []
    negative_class = []
    for i in range(num_classes):
        positive_class.append(0)
        negative_class.append(0)

    positive_num, negative_num = 0, 0
    for i in range(num_items):
        for j in range(len(label_matrix_obs[i])):
            if label_matrix_obs[i][j] == 1:
                positive_num += 1
                positive_class[j] += 1
            elif label_matrix_obs[i][j] == -1:
                negative_num += 1
                negative_class[j] += 1

    postive_ratio = positive_num / num_items
    negative_ratio = negative_num / num_items

    print("Positive ratio: {}, Negative ratio: {}".format(postive_ratio, negative_ratio))
    print("Positive class num : {}".format(positive_class))
    print("Negative class num : {}".format(negative_class))

    return label_matrix_obs


base_path = os.path.join('../datasets/{}'.format(args.dataset))

for phase in ['train', 'val']:
    # load ground truth binary label matrix:
    label_matrix = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
    print(len(label_matrix))
    assert np.max(label_matrix) == 1
    assert np.min(label_matrix) == 0

    # # convert label matrix to -1 / +1 format:
    # label_matrix[label_matrix == 0] = -1

    # choose observed labels, resulting in -1 / 0 / +1 format:
    label_matrix_obs = observe_uniform(label_matrix, args.num_labeled)

    # # save observed labels:
    # np.save(os.path.join(base_path, 'formatted_{}_labels_obs.npy'.format(phase)), label_matrix_obs)
