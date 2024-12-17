import numpy as np
import os
import argparse

import torch

pp = argparse.ArgumentParser(description='')
pp.add_argument('--dataset', type=str, choices=['voc', 'coco', 'nus', 'cub'], default='cub')
pp.add_argument('--num-label', type=int, default=1, help='number of positive or negative labels per image')
pp.add_argument('--seed', type=int, default=1200, help='random seed')
args = pp.parse_args()


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


def observe_uniform2(label_matrix, num_label, rng):
    '''
    label_matrix: binary (-1/+1) label matrix with shape num_items x num_classes
    num_pos: number of positive labels to observe for each item
    num_neg: number of negative labes to observe for each item
    rng: random number generator to use
    '''

    # check the observation parameters:
    assert num_label >= -1

    # check that label_matrix is a binary numpy array:
    assert type(label_matrix) is np.ndarray
    label_values = np.unique(label_matrix)
    assert len(label_values) == 2
    assert -1 in label_values
    assert 1 in label_values

    # apply uniform observation process:
    num_items, num_classes = np.shape(label_matrix)
    label_matrix_obs = np.zeros_like(label_matrix)
    for i in range(num_items):
        if i % 2 == 0:
            idx_pos = get_random_label_indices(label_matrix[i, :], 1.0, num_label, rng)
            label_matrix_obs[i, idx_pos] = 1.0
        else:
            idx_neg = get_random_label_indices(label_matrix[i, :], -1.0, num_label, rng)
            label_matrix_obs[i, idx_neg] = -1.0

    return label_matrix_obs


def observe_uniform(label_matrix, num_label, rng):
    '''
    label_matrix: binary (-1/+1) label matrix with shape num_items x num_classes
    num_pos: number of positive labels to observe for each item
    num_neg: number of negative labes to observe for each item
    rng: random number generator to use
    '''

    # check the observation parameters:
    assert num_label >= -1

    # check that label_matrix is a binary numpy array:
    assert type(label_matrix) is np.ndarray
    label_values = np.unique(label_matrix)
    assert len(label_values) == 2
    assert -1 in label_values
    assert 1 in label_values

    # apply uniform observation process:
    num_items, num_classes = np.shape(label_matrix)
    label_matrix_obs = np.zeros_like(label_matrix)
    positive_index = []
    negative_index = []
    for i in range(num_items):
        index = rng.randint(0, num_classes)
        if label_matrix[i, index] == 1.0:
            label_matrix_obs[i, index] = 1.0
            positive_index.append(i)
        else:
            label_matrix_obs[i, index] = -1.0
            negative_index.append(i)

    # positive_num = 0.5 * (label_matrix_obs.sum() + num_items)
    # negative_num = num_items - positive_num

    positive_label = torch.index_select(torch.tensor(label_matrix_obs), dim=0, index=torch.tensor(positive_index))
    negative_label = torch.index_select(torch.tensor(label_matrix_obs), dim=0, index=torch.tensor(negative_index))
    positive_num = positive_label.sum()
    negative_num = - negative_label.sum()

    postive_ratio = positive_num / num_items
    negative_ratio = negative_num / num_items

    print("Positive ratio: {}, Negative ratio: {}".format(postive_ratio, negative_ratio))
    print("Positive class num : {}".format(torch.sum(positive_label, dim=0)))
    print("Zero Positive class num".format())
    print("Negative class num : {}".format(torch.sum(negative_label, dim=0)))

    return label_matrix_obs


base_path = os.path.join('D:\datasets\mll_datasets/{}'.format(args.dataset))

for phase in ['train', 'val']:
    # load ground truth binary label matrix:
    label_matrix = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
    print(len(label_matrix))
    assert np.max(label_matrix) == 1
    assert np.min(label_matrix) == 0

    # convert label matrix to -1 / +1 format:
    label_matrix[label_matrix == 0] = -1

    # choose observed labels, resulting in -1 / 0 / +1 format:
    rng = np.random.RandomState(args.seed)
    label_matrix_obs = observe_uniform(label_matrix, args.num_label, rng)

    # save observed labels:
    np.save(os.path.join(base_path, 'formatted_{}_labels_obs.npy'.format(phase)), label_matrix_obs)
