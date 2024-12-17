
from clip import clip
from torch.nn import Parameter
from torch.nn.functional import relu, sigmoid
import torch.nn.functional as F
import torch
import numpy as np

from losses import rc_loss, rc_loss2



def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


single_template = ["a photo of a {}."]

multiple_templates = [
    "There is {article} {} in the scene, similar to {}, {}, and {}",
    "There is the {} in the scene, similar to {}, {}, and {}",
    "a photo of {article} {} in the scene, similar to {}, {}, and {}",
    "a photo of the {} in the scene, similar to {}, {}, and {}",
    "a photo of one {} in the scene, similar to {}, {}, and {}",
    "itap of {article} {}, similar to {}, {}, and {}",
    "itap of my {}, similar to {}, {}, and {}",  # itap: I took a picture of
    "itap of the {}, similar to {}, {}, and {}",
    "a photo of {article} {}, similar to {}, {}, and {}",
    "a photo of my {}, similar to {}, {}, and {}",
    "a photo of the {}, similar to {}, {}, and {}",
    "a photo of one {}, similar to {}, {}, and {}",
    "a photo of many {}, similar to {}, {}, and {}",
    "a good photo of {article} {}, similar to {}, {}, and {}",
    "a good photo of the {}, similar to {}, {}, and {}",
    "a bad photo of {article} {}, similar to {}, {}, and {}",
    "a bad photo of the {}, similar to {}, {}, and {}",
    "a photo of a nice {}, similar to {}, {}, and {}",
    "a photo of the nice {}, similar to {}, {}, and {}",
    "a photo of a cool {}, similar to {}, {}, and {}",
    "a photo of the cool {}, similar to {}, {}, and {}",
    "a photo of a weird {}, similar to {}, {}, and {}",
    "a photo of the weird {}, similar to {}, {}, and {}",
    "a photo of a small {}, similar to {}, {}, and {}",
    "a photo of the small {}, similar to {}, {}, and {}",
    "a photo of a large {}, similar to {}, {}, and {}",
    "a photo of the large {}, similar to {}, {}, and {}",
    "a photo of a clean {}, similar to {}, {}, and {}",
    "a photo of the clean {}, similar to {}, {}, and {}",
    "a photo of a dirty {}, similar to {}, {}, and {}",
    "a photo of the dirty {}, similar to {}, {}, and {}",
    "a bright photo of {article} {}, similar to {}, {}, and {}",
    "a bright photo of the {}, similar to {}, {}, and {}",
    "a dark photo of {article} {}, similar to {}, {}, and {}",
    "a dark photo of the {}, similar to {}, {}, and {}",
    "a photo of a hard to see {}, similar to {}, {}, and {}",
    "a photo of the hard to see {}, similar to {}, {}, and {}",
    "a low resolution photo of {article} {}, similar to {}, {}, and {}",
    "a low resolution photo of the {}, similar to {}, {}, and {}",
    "a cropped photo of {article} {}, similar to {}, {}, and {}",
    "a cropped photo of the {}, similar to {}, {}, and {}",
    "a close-up photo of {article} {}, similar to {}, {}, and {}",
    "a close-up photo of the {}, similar to {}, {}, and {}",
    "a jpeg corrupted photo of {article} {}, similar to {}, {}, and {}",
    "a jpeg corrupted photo of the {}, similar to {}, {}, and {}",
    "a blurry photo of {article} {}, similar to {}, {}, and {}",
    "a blurry photo of the {}, similar to {}, {}, and {}",
    "a pixelated photo of {article} {}, similar to {}, {}, and {}",
    "a pixelated photo of the {}, similar to {}, {}, and {}",
    "a black and white photo of the {}, similar to {}, {}, and {}",
    "a black and white photo of {article} {}, similar to {}, {}, and {}",
    "a plastic {}, similar to {}, {}, and {}",
    "the plastic {}, similar to {}, {}, and {}",
    "a toy {}, similar to {}, {}, and {}",
    "the toy {}, similar to {}, {}, and {}",
    "a plushie {}, similar to {}, {}, and {}",
    "the plushie {}, similar to {}, {}, and {}",
    "a cartoon {}, similar to {}, {}, and {}",
    "the cartoon {}, similar to {}, {}, and {}",
    "an embroidered {}, similar to {}, {}, and {}",
    "the embroidered {}, similar to {}, {}, and {}",
    "a painting of the {}, similar to {}, {}, and {}",
    "a painting of a {}, similar to {}, {}, and {}",
]

multiple_templates_dynamic_k= [
    "There is {article} {} in the scene, similar to",
    "There is the {} in the scene, similar to",
    "a photo of {article} {} in the scene, similar to",
    "a photo of the {} in the scene, similar to",
    "a photo of one {} in the scene, similar to",
    "itap of {article} {}, similar to",
    "itap of my {}, similar to",  # itap: I took a picture of
    "itap of the {}, similar to",
    "a photo of {article} {}, similar to",
    "a photo of my {}, similar to",
    "a photo of the {}, similar to",
    "a photo of one {}, similar to",
    "a photo of many {}, similar to",
    "a good photo of {article} {}, similar to",
    "a good photo of the {}, similar to",
    "a bad photo of {article} {}, similar to",
    "a bad photo of the {}, similar to",
    "a photo of a nice {}, similar to",
    "a photo of the nice {}, similar to",
    "a photo of a cool {}, similar to",
    "a photo of the cool {}, similar to",
    "a photo of a weird {}, similar to",
    "a photo of the weird {}, similar to",
    "a photo of a small {}, similar to",
    "a photo of the small {}, similar to",
    "a photo of a large {}, similar to",
    "a photo of the large {}, similar to",
    "a photo of a clean {}, similar to",
    "a photo of the clean {}, similar to",
    "a photo of a dirty {}, similar to",
    "a photo of the dirty {}, similar to",
    "a bright photo of {article} {}, similar to",
    "a bright photo of the {}, similar to",
    "a dark photo of {article} {}, similar to",
    "a dark photo of the {}, similar to",
    "a photo of a hard to see {}, similar to",
    "a photo of the hard to see {}, similar to",
    "a low resolution photo of {article} {}, similar to",
    "a low resolution photo of the {}, similar to",
    "a cropped photo of {article} {}, similar to",
    "a cropped photo of the {}, similar to",
    "a close-up photo of {article} {}, similar to",
    "a close-up photo of the {}, similar to",
    "a jpeg corrupted photo of {article} {}, similar to",
    "a jpeg corrupted photo of the {}, similar to",
    "a blurry photo of {article} {}, similar to",
    "a blurry photo of the {}, similar to",
    "a pixelated photo of {article} {}, similar to",
    "a pixelated photo of the {}, similar to",
    "a black white photo of the {}, similar to",
    "a black white photo of {article} {}, similar to",
    "a plastic {}, similar to",
    "the plastic {}, similar to",
    "a toy {}, similar to",
    "the toy {}, similar to",
    "a plushie {}, similar to",
    "the plushie {}, similar to",
    "a cartoon {}, similar to",
    "the cartoon {}, similar to",
    "an embroidered {}, similar to",
    "the embroidered {}, similar to",
    "a painting of the {}, similar to",
    "a painting of a {}, similar to",
]

similar_list2 = [
    ['airplane', 'jet', 'glider'],
    ['bike', 'cycle', 'unicycle'],
    ['fowl', 'avian', 'feathered'],
    ['ship', 'yacht', 'kayak'],
    ['flask', 'jar', 'pitcher'],
    ['coach', 'tram', 'van'],
    ['auto', 'vehicle', 'motorcar'],
    ['feline', 'kitty', 'moggy'],
    ['stool', 'seat', 'throne'],
    ['bull', 'heifer', 'beef cow'],
    ['banquet table', 'board', 'picnic table'],
    ['hound', 'mutt', 'pooch'],
    ['steed', 'pony', 'unicorn'],
    ['motorcycle', 'moto', 'hog'],
    ['human', 'individual', 'man'],
    ['horticultural plant', 'flowering plant', 'vascular plant'],
    ['ewe', 'lamb', 'ram'],
    ['couch', 'settee', 'love seat'],
    ['locomotive', 'railway train', 'railcar'],
    ['television set', 'video monitor', 'cathode ray tube']
]

def get_similar_list(categories, k=3):
    similar_list = []

    model, _ = clip.load('ViT-B/16')
    model = model.cuda()
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).cuda()
    categories_text_features = model.encode_text(text_inputs)
    categories_text_features /= categories_text_features.norm(dim=-1, keepdim=True)

    ram_taglist_file = "../ram/data/ram_tag_list.txt"
    with open(ram_taglist_file, "r", encoding="utf-8") as f:
        ram_taglist = [line.strip() for line in f]
    ram_text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in ram_taglist]).cuda()
    with torch.no_grad():
        ram_text_features = model.encode_text(ram_text_inputs)
    ram_text_features /= ram_text_features.norm(dim=-1, keepdim=True)

    value_list = []

    for i in range(len(categories)):
        similar_list.append([])
        value_list.append([])
        similarity = (100.0 * categories_text_features[i] @ ram_text_features.T).softmax(dim=-1)
        values, indices = similarity.topk(k)
        for value, index in zip(values, indices):
            similar_list[i].append(ram_taglist[index])
            value_list[i].append(value.item())
    # for i in range(len(similar_list)):
    #     print(similar_list[i])
    #     print(value_list[i])
    return similar_list


def get_similar_labels_str(similar_labels):
    if len(similar_labels) == 1:
        similar_labels_str = ' ' + similar_labels[0] + '.'
    elif len(similar_labels) == 2:
        similar_labels_str = ' ' + similar_labels[0] + ', and ' + similar_labels[1] + '.'
    else:
        other_elements = similar_labels[:-2]
        other_str = ', '.join(other_elements)
        last_two_elements = similar_labels[-2:]
        last_two_str = ', and '.join(last_two_elements)
        similar_labels_str = ' ' + other_str + ', ' + last_two_str + '.'

    return similar_labels_str

def build_openset_label_embedding(categories, similar_list):
    # print("Creating pretrained CLIP model")

    model, _ = clip.load("ViT-B/16")

    templates = multiple_templates_dynamic_k

    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        openset_label_embedding = []
        for i in range(len(categories)):
            category = categories[i]
            similar_labels = similar_list[i]
            texts = [
                template.format(
                    processed_name(category),
                    article=article(category)
                )
                for template in templates
            ]
            if len(similar_labels) > 0:
                similar_labels_str = get_similar_labels_str(similar_labels)

                texts = [
                    text + similar_labels_str for text in texts
                ]

            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            openset_label_embedding.append(text_embedding)
        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)
        if run_on_gpu:
            openset_label_embedding = openset_label_embedding.cuda()

    openset_label_embedding = openset_label_embedding.t()
    return openset_label_embedding, categories



import json
from tqdm import tqdm

def build_openset_llm_label_embedding(llm_tag_des):
    print("Creating pretrained CLIP model")
    model, _ = clip.load("ViT-B/16")
    llm_tag_des = llm_tag_des
    categories = []

    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        openset_label_embedding = []
        for item in tqdm(llm_tag_des):
            category = list(item.keys())[0]
            des = list(item.values())[0]

            categories.append(category)

            texts = clip.tokenize(des, truncate=True)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            openset_label_embedding.append(text_embedding)
        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)
        if run_on_gpu:
            openset_label_embedding = openset_label_embedding.cuda()

    openset_label_embedding = openset_label_embedding.t()
    return openset_label_embedding, categories


def get_num_similar_list(similar_list, num):
    new_similar_list = []
    for i in range(len(similar_list)):
        new_similar_list.append([])
        for j in range(num):
            new_similar_list[i].append(similar_list[i][j])
    return new_similar_list


def update_label_embeds2(origin_similar_list, similar_list, pos_imgs, pos_labels, neg_imgs, neg_labels,
                        taglist, model, ram_model):
    device = pos_imgs.device
    K = []
    update_similar_list = similar_list.copy()
    for i in range(len(taglist)):
        K.append(0)
        loss_min = 10000
        for j in range(3):
            similar_list[i] = origin_similar_list[i][:j+1]
            origin_label_embed, _ = build_openset_label_embedding(taglist, similar_list)
            label_embed = Parameter(origin_label_embed.float(), requires_grad=False)
            pos_image_embeds = ram_model.image_proj(ram_model.visual_encoder(pos_imgs))
            pos_image_embeds = pos_image_embeds.to(device)
            pos_image_atts = torch.ones(
                pos_image_embeds.size()[:-1], dtype=torch.long).to(device)
            pos_label_embed = relu(ram_model.wordvec_proj(label_embed)).unsqueeze(0) \
                .repeat(pos_imgs.shape[0], 1, 1)
            pos_label_embed = pos_label_embed.to(device)
            pos_tagging_embed, _ = ram_model.tagging_head(
                encoder_embeds=pos_label_embed,
                encoder_hidden_states=pos_image_embeds,
                encoder_attention_mask=pos_image_atts,
                return_dict=False,
                mode='tagging',
            )

            pos_ram_logits = ram_model.fc(pos_tagging_embed).squeeze(-1)
            pos_logits = model(pos_tagging_embed)

            neg_imgs = neg_imgs.to(device)
            neg_labels = neg_labels.to(device)
            neg_labels = torch.tensor(neg_labels, dtype=torch.float32)

            neg_image_embeds = ram_model.image_proj(ram_model.visual_encoder(neg_imgs))
            neg_image_embeds = neg_image_embeds.to(device)
            neg_image_atts = torch.ones(
                neg_image_embeds.size()[:-1], dtype=torch.long).to(device)
            neg_label_embed = relu(ram_model.wordvec_proj(label_embed)).unsqueeze(0) \
                .repeat(neg_imgs.shape[0], 1, 1)
            neg_label_embed = neg_label_embed.to(device)
            neg_tagging_embed, _ = ram_model.tagging_head(
                encoder_embeds=neg_label_embed,
                encoder_hidden_states=neg_image_embeds,
                encoder_attention_mask=neg_image_atts,
                return_dict=False,
                mode='tagging',
            )

            neg_ram_logits = ram_model.fc(neg_tagging_embed).squeeze(-1)
            neg_logits = model(neg_tagging_embed)

            new_loss = rc_loss2(pos_logits, pos_ram_logits, pos_labels, neg_logits, neg_ram_logits, neg_labels, 0)
            if new_loss.item() < loss_min:
                loss_min = new_loss.item()
                update_similar_list[i] = origin_similar_list[i][:j+1]
                K[i] = j + 1
        similar_list[i] = update_similar_list[i]
    label_embed, _ = build_openset_label_embedding(taglist, update_similar_list)
    ram_model.label_embed = Parameter(label_embed.float(), requires_grad=False)
    print("Similar list, best K: {}".format(','.join([str(x) for x in K])))
    return update_similar_list


def update_label_embeds(origin_similar_list, similar_list, imgs, labels, taglist, model, ram_model, loss_value):
    device = imgs.device
    similar_list = []
    for i in range(len(taglist)):
        similar_list.append([])
    K = []
    loss_min = loss_value
    update_similar_list = similar_list.copy()
    for i in range(len(taglist)):
        K.append(0)
        for j in range(1):
            similar_list[i] = origin_similar_list[i][:j+1]
            origin_label_embed, _ = build_openset_label_embedding(taglist, similar_list)
            label_embed = Parameter(origin_label_embed.float(), requires_grad=False)
            image_embeds = ram_model.image_proj(ram_model.visual_encoder(imgs))
            image_embeds = image_embeds.to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            label_embed = relu(ram_model.wordvec_proj(label_embed)).unsqueeze(0).repeat(imgs.shape[0], 1, 1)
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
            new_loss = rc_loss(logits, ram_logits, labels)
            if new_loss.item() < loss_min:
                loss_min = new_loss.item()
                update_similar_list[i] = origin_similar_list[i][:j+1]
                K[i] = j + 1
        similar_list[i] = update_similar_list[i]
    label_embed, _ = build_openset_label_embedding(taglist, update_similar_list)
    ram_model.label_embed = Parameter(label_embed.float(), requires_grad=False)
    print("Similar list, best K: {}".format(','.join([str(x) for x in K])))
    return update_similar_list



