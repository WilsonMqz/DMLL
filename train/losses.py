import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-5
ls_coef = 0.1
lookup = {'expected_num_pos': {
    'voc': 1.5,
    'coco': 2.9,
    'nus': 1.9,
    'cub': 31.4
},
}


def neg_log(x):
    LOG_EPSILON = 1e-5
    return - torch.log(x + LOG_EPSILON)


def inverse_sigmoid(p):
    epsilon = 1e-5
    p = np.minimum(p, 1 - epsilon)
    p = np.maximum(p, epsilon)
    return np.log(p / (1 - p))


def expected_positive_regularizer(preds, expected_num_pos, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos) ** 2
    else:
        raise NotImplementedError
    return reg


def loss_bce(preds, observed_labels):
    # input validation:
    assert not torch.any(observed_labels == -1)
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    return loss_mtx


def loss_an(preds, observed_labels):
    preds = torch.sigmoid(preds)
    observed_labels[observed_labels == -1] = 0
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    return loss_mtx.mean()


def loss_an_ls(preds, observed_labels):
    preds = torch.sigmoid(preds)
    observed_labels[observed_labels == -1] = 0
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - ls_coef) * neg_log(preds[observed_labels == 1]) + ls_coef * neg_log(
        1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - ls_coef) * neg_log(1.0 - preds[observed_labels == 0]) + ls_coef * neg_log(
        preds[observed_labels == 0])
    return loss_mtx.mean()


def loss_wan(preds, observed_labels, num_classes):
    preds = torch.sigmoid(preds)
    observed_labels[observed_labels == -1] = 0
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0]) / float(num_classes - 1)
    return loss_mtx.mean()


def loss_epr(preds, observed_labels):
    preds = torch.sigmoid(preds)
    observed_labels[observed_labels == -1] = 0
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss w.r.t. observed positives:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    return loss_mtx.mean()


def get_estimated_labels(observed_labels, num_classes):
    num_examples = int(np.shape(observed_labels)[0])
    observed_label_matrix = np.array(observed_labels.cpu()).astype(np.int8)
    total_pos = np.sum(observed_label_matrix == 1)
    total_neg = np.sum(observed_label_matrix == -1)
    # print('observed positives: {} total, {:.1f} per example on average'.format(total_pos, total_pos / num_examples))
    # print('observed negatives: {} total, {:.1f} per example on average'.format(total_neg, total_neg / num_examples))

    # initialize unobserved labels:
    w = 0.1
    q = inverse_sigmoid(0.5 + w)
    param_mtx = q * (2 * torch.rand(num_examples, num_classes) - 1)

    # initialize observed positive labels:
    init_logit_pos = inverse_sigmoid(0.995)
    idx_pos = torch.from_numpy((observed_label_matrix == 1).astype(np.bool_))
    param_mtx[idx_pos] = init_logit_pos

    # initialize observed negative labels:
    init_logit_neg = inverse_sigmoid(0.005)
    idx_neg = torch.from_numpy((observed_label_matrix == -1).astype(np.bool_))
    param_mtx[idx_neg] = init_logit_neg
    return torch.sigmoid(param_mtx)


def loss_role(preds, observed_labels, dataset, num_classes):
    preds = torch.sigmoid(preds)
    observed_labels[observed_labels == -1] = 0
    expected_num_pos = lookup['expected_num_pos'][dataset]
    estimated_labels = get_estimated_labels(observed_labels, num_classes)
    estimated_labels = estimated_labels.to(preds.device)
    # input validation:
    assert torch.min(observed_labels) >= 0
    # (image classifier) compute loss w.r.t. observed positives:
    loss_mtx_pos_1 = torch.zeros_like(observed_labels)
    loss_mtx_pos_1[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # (image classifier) compute loss w.r.t. label estimator outputs:
    estimated_labels_detached = estimated_labels.detach()
    loss_mtx_cross_1 = estimated_labels_detached * neg_log(preds) + (1.0 - estimated_labels_detached) * neg_log(
        1.0 - preds)
    # (image classifier) compute regularizer:
    reg_1 = expected_positive_regularizer(preds, expected_num_pos, norm='2') / (num_classes ** 2)
    # (label estimator) compute loss w.r.t. observed positives:
    loss_mtx_pos_2 = torch.zeros_like(observed_labels)
    loss_mtx_pos_2[observed_labels == 1] = neg_log(estimated_labels[observed_labels == 1])
    # (label estimator) compute loss w.r.t. image classifier outputs:
    preds_detached = preds.detach()
    loss_mtx_cross_2 = preds_detached * neg_log(estimated_labels) + (1.0 - preds_detached) * neg_log(
        1.0 - estimated_labels)
    # (label estimator) compute regularizer:
    reg_2 = expected_positive_regularizer(estimated_labels, expected_num_pos, norm='2') / (num_classes ** 2)
    # compute final loss matrix:
    reg_loss = 0.5 * (reg_1 + reg_2)
    loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
    loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)

    return loss_mtx.mean(), reg_loss


'''
ECCV 2022
'''


def loss_EM(preds, observed_labels, dataset):
    preds = torch.sigmoid(preds)
    if dataset == 'voc':
        alpha = 0.2
    elif dataset == 'cub':
        alpha = 0.01
    elif dataset == 'nus':
        alpha = 0.1
    elif dataset == 'coco':
        alpha = 0.1

    observed_labels[observed_labels == -1] = 0

    # input validation:
    assert torch.min(observed_labels) >= 0

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -alpha * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
    )

    return loss_mtx.mean()


'''
ECCV 2022
'''


def loss_EM_APL(preds, observed_labels, dataset):
    preds = torch.sigmoid(preds)
    if dataset == 'voc':
        alpha = 0.2
        beta = 0.02
    elif dataset == 'cub':
        alpha = 0.01
        beta = 0.4
    elif dataset == 'nus':
        alpha = 0.1
        beta = 0.2
    elif dataset == 'coco':
        alpha = 0.1
        beta = 0.9

    # input validation:
    assert torch.min(observed_labels) >= -1

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -alpha * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
    )

    soft_label = -observed_labels[observed_labels < 0]
    loss_mtx[observed_labels < 0] = beta * (
            soft_label * neg_log(preds[observed_labels < 0]) +
            (1 - soft_label) * neg_log(1 - preds[observed_labels < 0])
    )
    return loss_mtx.mean()


def loss_LL_an(logits, observed_labels):
    logits = torch.sigmoid(logits)
    observed_labels[observed_labels == -1] = 0
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_matrix = F.binary_cross_entropy_with_logits(logits, observed_labels, reduction='none')
    corrected_loss_matrix = F.binary_cross_entropy_with_logits(logits, torch.logical_not(observed_labels).float(),
                                                               reduction='none')
    return loss_matrix, corrected_loss_matrix


'''
cvpr 2022
'''


def loss_LL(preds, label_vec, method, epoch):  # "preds" are actually logits (not sigmoid activated !)
    # preds = torch.sigmoid(preds)
    assert preds.dim() == 2

    batch_size = int(preds.size(0))
    num_classes = int(preds.size(1))

    unobserved_mask = (label_vec == 0)

    # compute loss for each image and class:
    loss_matrix, corrected_loss_matrix = loss_LL_an(preds, label_vec)

    # delta_rel = 0.1
    # if delta_rel != 0:
    #     delta_rel /= 100
    #
    # clean_rate = 1 - (epoch + 1) * delta_rel
    #
    # if method == 'LL-Cp':
    #     k = math.ceil(batch_size * num_classes * delta_rel)
    # else:
    #     k = math.ceil(batch_size * num_classes * (1 - clean_rate))

    k = math.ceil(batch_size * num_classes * (epoch * 0.002))

    unobserved_loss = unobserved_mask.bool() * loss_matrix
    topk = torch.topk(unobserved_loss.flatten(), k)
    topk_lossvalue = topk.values[-1]
    correction_idx = torch.where(unobserved_loss > topk_lossvalue)
    if method in ['LL-Ct', 'LL-Cp']:
        final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, corrected_loss_matrix)
    else:
        zero_loss_matrix = torch.zeros_like(loss_matrix)
        final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)

    main_loss = final_loss_matrix.mean()

    return main_loss, correction_idx


def loss_Scob_bce(pred: Tensor, labels: Tensor, only_pos: bool = False) -> Tensor:
    num_classes = labels.size(1)

    return (
            -(
                    (pred + EPS).log() * labels
                    + (-pred + 1.0 + EPS).log() * (-labels + 1.0 + EPS)
            ).sum(dim=1)
            / num_classes
    )


def loss_Scob_epr(
        pred: Tensor, strong_labels: Tensor, expected_num_pos_labels: float
) -> Tensor:
    batch_size = pred.size(0)
    label_size = pred.size(1)

    first_part = loss_Scob_bce(pred, strong_labels).sum(0) / batch_size
    second_part = ((pred.sum(1).mean(0) - expected_num_pos_labels) / label_size) ** 2

    return first_part + second_part


'''
Scob 2023 IJCV
'''


def loss_Scob(pred_targets, observed_targets, dataset, num_classes):
    pred_targets = torch.sigmoid(pred_targets)
    expected_num_pos_labels = lookup['expected_num_pos'][dataset]
    esti_targets = get_estimated_labels(observed_targets, num_classes)
    esti_targets = esti_targets.to(pred_targets.device)

    # make sure all labels are positive
    observed_targets[observed_targets == -1] = 0
    assert torch.min(observed_targets) >= 0

    batch_size = pred_targets.size(0)

    # freeze estimated targets
    esti_targets_detach = esti_targets.detach()
    loss_1_1 = loss_Scob_bce(pred_targets, esti_targets_detach).sum(
        dim=0
    ) / batch_size + loss_Scob_epr(pred_targets, observed_targets, expected_num_pos_labels)
    loss_1 = loss_1_1
    # freeze predicted targets
    pred_targets_detach = pred_targets.detach()
    loss_2 = loss_Scob_bce(esti_targets, pred_targets_detach).sum(
        dim=0
    ) / batch_size + loss_Scob_epr(esti_targets, observed_targets, expected_num_pos_labels)
    return (loss_1 + loss_2) / 2


'''
VLPL 2023 arbix
'''


def loss_VLPL(preds, pseudo_labels, observed_labels, dataset):
    preds = torch.sigmoid(preds)
    if dataset == 'voc':
        alpha = 0.2
    elif dataset == 'cub':
        alpha = 0.01
    elif dataset == 'nus':
        alpha = 0.1
    elif dataset == 'coco':
        alpha = 0.1

    beta_pos = 0.7

    final_labels = torch.where(observed_labels == 0, pseudo_labels, observed_labels)
    # input validation:
    assert torch.min(final_labels) >= -1

    loss_mtx = torch.zeros_like(preds)
    #####
    # observed positive label
    #####
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])

    #####
    # Unknown Labels
    #####
    loss_mtx[final_labels == 0] = -alpha * (preds[final_labels == 0] * neg_log(preds[final_labels == 0]) + (
            1 - preds[final_labels == 0]) * neg_log(1 - preds[final_labels == 0]))

    #####
    # Pseudo-Label
    #####
    # positive pseudo-label
    mask_pos = (observed_labels == 0) & (pseudo_labels == 1)
    # print(mask)
    loss_mtx[mask_pos] = beta_pos * (0.9 * neg_log(preds[mask_pos]) + 0.1 * neg_log(1 - preds[mask_pos]))

    return loss_mtx.mean()


'''
PLCSL CVPR 2022 partial
'''


def loss_PLCSL(logits, targets):
    alpha_pos = 1
    alpha_neg = 1
    alpha_unann = 1
    gamma_pos = 0
    gamma_neg = 1
    gamma_unann = 4
    # Positive, Negative and Un-annotated indexes
    targets_pos = (targets == 1).float()
    targets_neg = (targets == -1).float()
    targets_unann = (targets == 0).float()

    # Activation
    xs_pos = torch.sigmoid(logits)
    xs_neg = 1.0 - xs_pos

    # Loss calculation
    BCE_pos = alpha_pos * targets_pos * torch.log(torch.clamp(xs_pos, min=1e-8))
    BCE_neg = alpha_neg * targets_neg * torch.log(torch.clamp(xs_neg, min=1e-8))
    BCE_unann = alpha_unann * targets_unann * torch.log(torch.clamp(xs_neg, min=1e-8))

    BCE_loss = BCE_pos + BCE_neg + BCE_unann

    # Adding asymmetric gamma weights
    with torch.no_grad():
        asymmetric_w = torch.pow(1 - xs_pos * targets_pos - xs_neg * (targets_neg + targets_unann),
                                 gamma_pos * targets_pos + gamma_neg * targets_neg +
                                 gamma_unann * targets_unann)
    BCE_loss *= asymmetric_w

    return -BCE_loss.sum()


'''
MLCLL IJCAI 2023 complementary 
'''


def loss_GDF(outputs, labels, num_classes, device):
    com_labels = torch.zeros(len(labels), num_classes)
    com_labels[labels == -1] = 1
    # for j in range(len(labels)):
    #     com_labels[j, (int)(labels[j].item())] = 1
    sig = nn.Sigmoid()
    sig_outputs = sig(outputs)
    pos_outputs = 1 - com_labels
    pos_outputs = pos_outputs.to(device)
    neg_outputs = com_labels
    neg_outputs = neg_outputs.to(device)

    part_1 = -torch.sum(torch.log(sig_outputs + 1e-12) * pos_outputs, dim=1).mean()
    part_3 = -torch.sum(torch.log(1.0 - sig_outputs + 1e-12) * neg_outputs, dim=1).mean()
    ave_loss = part_1 + part_3
    return ave_loss


def loss_p(preds, origin_preds, labels):
    loss_mtx = origin_preds * neg_log(preds) + (1 - origin_preds) * neg_log(1 - preds)
    loss_mtx = torch.sum(loss_mtx, dim=1)
    # weights = origin_preds[labels == 1]
    # weights = 1.0 / (weights * labels.shape[1])
    # loss_mtx = weights * loss_mtx
    return loss_mtx.mean()


def loss_n(preds, origin_preds, labels):
    loss_mtx = origin_preds * neg_log(preds) + (1 - origin_preds) * neg_log(1 - preds)
    loss_mtx = torch.sum(loss_mtx, dim=1)
    # weights = origin_preds[labels == -1]
    # weights = 1.0 / ((1 - weights) * labels.shape[1])
    # loss_mtx = weights * loss_mtx
    return loss_mtx.mean()


def softmax_binary(logits, labels, epoch):
    temperature = 0.001

    logits_pos = logits[labels == 1]
    logits_neg = logits[labels == -1]

    logits_pos = logits_pos / temperature
    logits_pos = (np.exp(logits_pos.cpu().detach().numpy()) /
                  (np.exp(logits_pos.cpu().detach().numpy())
                   + np.exp(1 - logits_pos.cpu().detach().numpy())))
    logits_neg = logits_neg * temperature
    logits_neg = (np.exp(logits_neg.cpu().detach().numpy()) /
                  (np.exp(logits_neg.cpu().detach().numpy())
                   + np.exp(1 - logits_neg.cpu().detach().numpy())))

    logits[labels == 1] = torch.tensor(logits_pos, dtype=torch.float32).to(logits.device)
    logits[labels == -1] = torch.tensor(logits_neg, dtype=torch.float32).to(logits.device)


def softmax_binary_pos(logits, labels, temperature=0.1):
    logits_pos = logits[labels == 1]
    logits_pos = logits_pos / temperature
    logits_pos = (np.exp(logits_pos.cpu().detach().numpy()) /
                  (np.exp(logits_pos.cpu().detach().numpy())
                   + np.exp(1 - logits_pos.cpu().detach().numpy())))

    logits[labels == 1] = torch.tensor(logits_pos, dtype=torch.float32).to(logits.device)


def softmax_binary_neg(logits, labels, temperature=0.1):
    logits_neg = logits[labels == -1]
    logits_neg = logits_neg * temperature
    logits_neg = (np.exp(logits_neg.cpu().detach().numpy()) /
                  (np.exp(logits_neg.cpu().detach().numpy())
                   + np.exp(1 - logits_neg.cpu().detach().numpy())))
    logits[labels == -1] = torch.tensor(logits_neg, dtype=torch.float32).to(logits.device)


def rc_loss(logits, ram_logits, labels, epoch=0):
    device = logits.device
    sigmoid_logits = torch.sigmoid(logits)
    sigmoid_ram_logits = torch.sigmoid(ram_logits)

    # softmax_binary(sigmoid_ram_logits, labels, epoch)
    sigmoid_ram_logits[labels == 1] = 1
    sigmoid_ram_logits[labels == -1] = 0

    positive_index = []
    negative_index = []
    for i in range(labels.size()[0]):
        if labels[i].sum() == 1:
            positive_index.append(i)
        else:
            negative_index.append(i)
    positive_index = torch.tensor(positive_index).to(device)
    negative_index = torch.tensor(negative_index).to(device)
    labels_p = torch.index_select(labels, dim=0, index=positive_index)
    labels_n = torch.index_select(labels, dim=0, index=negative_index)
    sigmoid_logits_p = torch.index_select(sigmoid_logits, dim=0, index=positive_index)
    sigmoid_logits_n = torch.index_select(sigmoid_logits, dim=0, index=negative_index)
    sigmoid_ram_logits_p = torch.index_select(sigmoid_ram_logits, dim=0, index=positive_index)
    sigmoid_ram_logits_n = torch.index_select(sigmoid_ram_logits, dim=0, index=negative_index)
    if len(labels_p) > 0:
        loss1 = loss_p(sigmoid_logits_p, sigmoid_ram_logits_p, labels_p)
        loss2 = loss_n(sigmoid_logits_n, sigmoid_ram_logits_n, labels_n)
        return loss1 + loss2
    else:
        loss2 = loss_n(sigmoid_logits_n, sigmoid_ram_logits_n, labels_n)
        return loss2


def rc_loss2(pos_logits, pos_ram_logits, pos_labels, neg_logits, neg_ram_logits, neg_labels, epoch):
    sigmoid_pos_logits = torch.sigmoid(pos_logits)
    sigmoid_pos_ram_logits = torch.sigmoid(pos_ram_logits)
    # softmax_binary_pos(sigmoid_pos_ram_logits, pos_labels)

    sigmoid_neg_logits = torch.sigmoid(neg_logits)
    sigmoid_neg_ram_logits = torch.sigmoid(neg_ram_logits)
    # softmax_binary_neg(sigmoid_neg_ram_logits, neg_labels)

    sigmoid_pos_ram_logits[pos_labels == 1] = 1
    sigmoid_neg_ram_logits[neg_labels == -1] = 0

    loss1 = loss_p(sigmoid_pos_logits, sigmoid_pos_ram_logits, pos_labels)
    loss2 = loss_n(sigmoid_neg_logits, sigmoid_neg_ram_logits, neg_labels)
    loss = loss1 + loss2
    return loss


def choose_loss_fn(logits, ram_logits, labels, taglist, loss_fn, dataset, epoch):
    num_classes = len(taglist)
    loss = None
    if loss_fn == 'an':
        loss = loss_an(logits, labels)
    elif loss_fn == 'an_ls':
        loss = loss_an_ls(logits, labels)
    elif loss_fn == 'wan':
        loss = loss_wan(logits, labels, num_classes)
    elif loss_fn == 'epr':
        loss = loss_epr(logits, labels)
    elif loss_fn == 'role':
        loss, _ = loss_role(logits, labels, dataset, num_classes)
    elif loss_fn == 'EM':
        loss = loss_EM(logits, labels, dataset)
    elif loss_fn == 'EM_APL':
        loss = loss_EM_APL(logits, labels, dataset)
    elif loss_fn == 'LL_R':
        loss, _ = loss_LL(logits, labels, 'LL-R', epoch)
    elif loss_fn == 'LL_Ct':
        loss, _ = loss_LL(logits, labels, 'LL-Ct', epoch)
    elif loss_fn == 'LL_Cp':
        loss, _ = loss_LL(logits, labels, 'LL-Cp', epoch)
    elif loss_fn == 'Scob':
        loss = loss_Scob(logits, labels, dataset, num_classes)
    elif loss_fn == 'VLPL':
        threshold = 0.8
        pseudo_labels = (torch.sigmoid(ram_logits.detach())).float()
        pseudo_labels = (pseudo_labels >= threshold).float()
        loss = loss_VLPL(logits, pseudo_labels, labels, dataset)
    elif loss_fn == 'PLCSL':
        loss = loss_PLCSL(logits, labels)
    elif loss_fn == 'GDF':
        loss = loss_GDF(logits, labels, num_classes, device=logits.device)

    return loss
