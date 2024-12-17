import sys

sys.path.append('..')
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, TextIO

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import relu, sigmoid
from tqdm import tqdm

from ram.models import ram
from ram.utils.metrics import get_mAP_sjml
from utils import step_lr_schedule
from dataset_loader import load_spml_datasets
from cls_model import MLP
from losses import rc_loss
from openset_SJML_utils_learning import build_openset_label_embedding, get_similar_list, update_label_embeds, \
    get_num_similar_list
from metrics import HammingLoss, CoverageLoss, AveragePrecisionLoss, RankingLossLoss, OneErrorLoss

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    # model
    parser.add_argument("--model-type", type=str, choices=("ram", "ram_plus"), default="ram")
    parser.add_argument("--checkpoint", type=str, default='../pretrained/ram_swin_large_14m.pth')
    parser.add_argument("--backbone", type=str, choices=("swin_l", "swin_b"), default="swin_l",
                        help="If `None`, will judge from `--model-type`")
    parser.add_argument("--open-set", type=bool, default=True,
                        help=(
                            "Treat all categories in the taglist file as "
                            "unseen and perform open-set classification. Only "
                            "works with RAM."
                        ))
    # threshold
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold", type=float, default=None,
                       help=(
                           "Use custom threshold for all classes. Mutually "
                           "exclusive with `--threshold-file`. If both "
                           "`--threshold` and `--threshold-file` is `None`, "
                           "will use a default threshold setting."
                       ))
    group.add_argument("--threshold-file", type=str, default=None,
                       help=(
                           "Use custom class-wise thresholds by providing a "
                           "text file. Each line is a float-type threshold, "
                           "following the order of the tags in taglist file. "
                           "See `ram/data/ram_tag_list_threshold.txt` as an "
                           "example. Mutually exclusive with `--threshold`. "
                           "If both `--threshold` and `--threshold-file` is "
                           "`None`, will use default threshold setting."
                       ))
    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./SJML_outputs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=100)
    # data
    parser.add_argument("--dataset", type=str, choices=("voc", "coco", "cub", "nus"), default="coco")
    parser.add_argument("--input-size", type=int, default=224)

    args = parser.parse_args()

    # post process and validity check
    args.model_type = args.model_type.lower()

    assert not (args.model_type == "tag2text" and args.open_set)

    if args.backbone is None:
        args.backbone = "swin_l" if args.model_type == "ram_plus" or args.model_type == "ram" else "swin_b"

    return args


def get_class_idxs(
        model_type: str,
        open_set: bool,
        taglist: List[str]
) -> Optional[List[int]]:
    """Get indices of required categories in the label system."""
    if model_type == "ram_plus" or model_type == "ram":
        if not open_set:
            model_taglist_file = "../ram/data/ram_tag_list.txt"
            with open(model_taglist_file, "r", encoding="utf-8") as f:
                model_taglist = [line.strip() for line in f]
            return [model_taglist.index(tag) for tag in taglist]
        else:
            return None
    else:  # for tag2text, we directly use tagid instead of text-form of tag.
        # here tagid equals to tag index.
        return [int(tag) for tag in taglist]


def load_thresholds(
        threshold: Optional[float],
        threshold_file: Optional[str],
        model_type: str,
        open_set: bool,
        class_idxs: List[int],
        num_classes: int,
) -> List[float]:
    """Decide what threshold(s) to use."""
    if not threshold_file and not threshold:  # use default
        if model_type == "ram_plus" or model_type == "ram":
            if not open_set:  # use class-wise tuned thresholds
                ram_threshold_file = "../ram/data/ram_tag_list_threshold.txt"
                with open(ram_threshold_file, "r", encoding="utf-8") as f:
                    idx2thre = {
                        idx: float(line.strip()) for idx, line in enumerate(f)
                    }
                    return [idx2thre[idx] for idx in class_idxs]
            else:
                return [0.5] * num_classes
        else:
            return [0.68] * num_classes
    elif threshold_file:
        with open(threshold_file, "r", encoding="utf-8") as f:
            thresholds = [float(line.strip()) for line in f]
        assert len(thresholds) == num_classes
        return thresholds
    else:
        return [threshold] * num_classes


def load_ram(
        backbone: str,
        checkpoint: str,
        input_size: int,
        taglist: List[str],
        open_set: bool,
        class_idxs: List[int]
):
    model = ram(pretrained=checkpoint, image_size=input_size, vit=backbone)
    # trim taglist for faster inference

    if open_set:
        print("Building tag embeddings ...")
        init_similar_list = get_num_similar_list(origin_similar_list, 0)
        label_embed, _ = build_openset_label_embedding(taglist, init_similar_list)
        print("Similar list, init K = 3")
        model.label_embed = Parameter(label_embed.float())
    else:
        model.label_embed = Parameter(model.label_embed[class_idxs, :])
    return model.to(device).eval()


def print_write(f: TextIO, s: str):
    print(s)
    f.write(s + "\n")


@torch.no_grad()
def test_model(ram_model, model, test_loader, taglist):
    ram_model.eval()
    model.eval()

    # inference
    final_logits = torch.empty(len(test_loader.dataset), len(taglist))
    targs = torch.empty(len(test_loader.dataset), len(taglist))
    pos = 0

    for (imgs, labels) in tqdm(test_loader, desc="Test"):
        labels = labels.to(device)
        image_embeds = ram_model.image_proj(ram_model.visual_encoder(imgs.to(device)))
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(device)
        label_embed = relu(ram_model.wordvec_proj(ram_model.label_embed)).unsqueeze(0) \
            .repeat(imgs.shape[0], 1, 1)
        tagging_embed, _ = ram_model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = model(tagging_embed)

        bs = imgs.shape[0]
        final_logits[pos:pos + bs, :] = sigmoid(logits).cpu()
        targs[pos:pos + bs, :] = labels.cpu()
        pos += bs

    # evaluate and record
    mAP, APs = get_mAP_sjml(final_logits.numpy(), targs.numpy(), taglist)
    Y_pred = torch.zeros_like(final_logits)
    Y_pred[final_logits > 0.5] = 1.0
    Hammingloss = HammingLoss(Y_pred, targs)
    Coverage = CoverageLoss(final_logits, targs)
    AveragePrecision = AveragePrecisionLoss(final_logits, targs)
    RankingLoss = RankingLossLoss(final_logits, targs)
    OneError = OneErrorLoss(final_logits, targs)

    return mAP, APs, Hammingloss, Coverage, AveragePrecision, RankingLoss, OneError


if __name__ == "__main__":
    args = parse_args()

    # fix random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # set up output paths
    output_dir = args.output_dir + "/" + args.dataset + "/" + str(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ap_file, summary_file, record_file = [
        output_dir + "/" + name for name in
        ("ap_SJML_RC_PromptLearning.txt", "summary_SJML_RC_PromptLearning.txt", "res_me_SJML_RC_PromptLearning.csv")
    ]
    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(f, "****************")
        for key in (
                "model_type", "backbone", "checkpoint", "open_set",
                "dataset", "input_size",
                "threshold", "threshold_file",
                "output_dir", "batch_size", "num_workers"
        ):
            print_write(f, f"{key}: {getattr(args, key)}")
        print_write(f, "****************")

    with open(record_file, 'a') as f:
        f.writelines("epoch,train_loss,mAP,HammingLoss,Coverage,AP,RankingLoss,OneError\n")

    # prepare data
    train_loader, info = load_spml_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="train",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    taglist, imglist, tag_des = \
        info["taglist"], info["imglist"], info["tag_des"]

    test_loader, _ = load_spml_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="val",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # get class idxs
    class_idxs = get_class_idxs(
        model_type=args.model_type,
        open_set=args.open_set,
        taglist=taglist
    )

    # set up threshold(s)
    thresholds = load_thresholds(
        threshold=args.threshold,
        threshold_file=args.threshold_file,
        model_type=args.model_type,
        open_set=args.open_set,
        class_idxs=class_idxs,
        num_classes=len(taglist)
    )

    origin_similar_list = get_similar_list(taglist, k=1)
    # similar_list = get_similar_list(taglist, k=1)
    similar_list = []
    for i in range(len(taglist)):
        similar_list.append([])

    if args.model_type == "ram":
        ram_model = load_ram(
            backbone=args.backbone,
            checkpoint=args.checkpoint,
            input_size=args.input_size,
            taglist=taglist,
            open_set=args.open_set,
            class_idxs=class_idxs
        )

    # freeze
    for params in ram_model.parameters():
        params.requires_grad = False

    num_classes = len(taglist)

    model = MLP(input_dim=768 * num_classes, output_dim=num_classes)

    l2_loss = nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.05)

    ram_model.to(device)
    model.to(device)

    best_mAP = 0.0
    update_label_embeds_flag = True
    # start training
    for epoch in range(args.epochs):

        torch.cuda.empty_cache()

        # update_label_embeds_flag = True

        step_lr_schedule(optimizer, epoch, init_lr=1e-2, min_lr=5e-5, decay_rate=0.9)
        lr = optimizer.param_groups[-1]['lr']

        idx = 0

        for (imgs, labels) in tqdm(train_loader, desc="Train"):

            optimizer.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = torch.tensor(labels, dtype=torch.float32)

            image_embeds = ram_model.image_proj(ram_model.visual_encoder(imgs))
            image_embeds = image_embeds.to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            label_embed = relu(ram_model.wordvec_proj(ram_model.label_embed)).unsqueeze(0).repeat(imgs.shape[0], 1, 1)
            label_embed = label_embed.to(device)
            tagging_embed, _ = ram_model.tagging_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
                mode='tagging',
            )

            ram_logits = ram_model.fc(tagging_embed).squeeze(-1)
            logits = model(tagging_embed)

            loss = rc_loss(logits, ram_logits.detach(), labels)

            loss.backward()
            optimizer.step()

            idx += 1

            # update similar_list and label_embeds
            if epoch > 0 and epoch % 5 == 0 and update_label_embeds_flag:
                print("update update similar_list and label_embeds")
                similar_list = update_label_embeds(origin_similar_list, similar_list, imgs, labels, taglist, model,
                                                   ram_model)
                update_label_embeds_flag = False

        # test and save checkpoint
        mAP, APs, Hammingloss, Coverage, AveragePrecision, RankingLoss, OneError = test_model(ram_model, model,
                                                                                              test_loader, taglist)

        if mAP >= best_mAP:
            save_obj = {
                'model': model.state_dict(),
                'epoch': epoch,
                'mAP': mAP,
                'APs': APs
            }

            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_best.pth'))

            with open(ap_file, "w", encoding="utf-8") as f:
                f.write("Tag,AP\n")
                for tag, AP in zip(taglist, APs):
                    f.write(f"{tag},{AP * 100.0:.2f}\n")

            with open(summary_file, "a", encoding="utf-8") as f:
                print_write(f, f"mAP: {mAP * 100.0}")

            best_mAP = mAP

        print("Epoch: {}| Lr: {:.4f}| Loss: {:.4f}| mAP: {:.3f}| Hamming: {:.4f}| Coverage: {:.4f}| AP: {:.4f}| "
              "Ranking: {:.4f}| OneError: {:.4f}".format(epoch, lr, loss.item(), mAP * 100.0, Hammingloss,
                                                         Coverage, AveragePrecision, RankingLoss, OneError))

        with open(record_file, 'a') as f:
            f.writelines("{},{:.4f},{:.3f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n"
                         .format(epoch, loss.item(), mAP * 100.0, Hammingloss, Coverage, AveragePrecision,
                                 RankingLoss, OneError))

    with open(record_file, 'a') as f:
        f.writelines("max,{:.3f}\n".format(best_mAP * 100.0))
