# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Models for Text and Image Composition."""
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import text_model
import torch_functions
from torch.autograd import Variable

class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=4.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()

    def extract_img_feature(self, imgs):
        raise NotImplementedError

    def extract_text_feature(self, text_query, use_bert):
        raise NotImplementedError

    def compose_img_text(self, imgs, text_query):
        raise NotImplementedError

    def compute_loss(self,
                     imgs_query,
                     input_ids, 
                     input_mask,
                     imgs_target,
                     args):
                     
        composed_feature= self.compose_img_text(imgs_query, input_ids, input_mask)
        target_feature = self.extract_img_feature(imgs_target)

        loss = 0
        if args.loss == "soft_triplet":
            for composed_feature_item, target_feature_item in zip(composed_feature, target_feature):
                loss += self.compute_soft_triplet_loss_(composed_feature_item, target_feature_item).cuda()
        elif args.loss == "batch_based_classification":
            for composed_feature_item, target_feature_item in zip(composed_feature, target_feature):
                loss += self.compute_batch_based_classification_loss_(composed_feature_item, target_feature_item).cuda()
                
        loss = (loss,)
        return loss

    def compute_soft_triplet_loss_(self, mod_img1, img2):
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert (triplets and len(triplets) < 2000)
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        return F.cross_entropy(x, labels)

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from models.bert_model import BertCrossLayer, BertAttention, BertSelfAttention, BertIntermediate, BertOutput
from models import swin_transformer as swin
from util import RegionLearner, init_weights, Pooler

class ComqueryFormer(ImgTextCompositionBase):
    def __init__(self, config):
        super().__init__(config)

        config = vars(config)
        self.local_region_ref = []
        self.local_region_tar = []
        for i in range(3):
            self.local_region_ref.append(RegionLearner(token_dim=config["hidden_size"], num_region=config["num_clusters"]))
            self.local_region_tar.append(RegionLearner(token_dim=config["hidden_size"], num_region=config["num_clusters"]))
        self.local_region_ref = nn.ModuleList(self.local_region_ref)
        self.local_region_tar = nn.ModuleList(self.local_region_tar)

        self.normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=4.0, learn_scale=True)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(init_weights)
        self.image_transformer = getattr(swin, config["vit"])(pretrained=True, config=config,)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.text_transformer = BertModel.from_pretrained(config['tokenizer'])


        self.text_linears = []
        for input_text_embed_size in config['input_text_embed_size']:
            self.text_linear = nn.Linear(input_text_embed_size, config['hidden_size'])
            self.text_linear.apply(init_weights)
            self.text_linears.append(self.text_linear)
        self.text_linears = nn.ModuleList(self.text_linears)

        self.image_linears = []
        for input_image_embed_size in config['input_image_embed_size']:
            self.image_linear = nn.Linear(input_image_embed_size, config['hidden_size'])
            self.image_linear.apply(init_weights)
            self.image_linears.append(self.image_linear)
        self.image_linears = nn.ModuleList(self.image_linears)

        bert_config = BertConfig(
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],)  
        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_layer'])])
        self.cross_modal_image_layers.apply(init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_layer'])])
        self.cross_modal_text_layers.apply(init_weights)

        self.image_pooler = Pooler(config["hidden_size"])
        self.image_pooler.apply(init_weights)
        self.text_pooler = Pooler(config["hidden_size"])
        self.text_pooler.apply(init_weights)

    def extract_img_feature(self, imgs):
        image_embeds_global = []
        image_features_local = []

        image_embeds = self.image_transformer.patch_embed(imgs)
        if self.image_transformer.absolute_pos_embed is not None:
            image_embeds = image_embeds + self.image_transformer.absolute_pos_embed
        image_embeds = self.image_transformer.pos_drop(image_embeds)

        layer_number = 0
        for layer in self.image_transformer.layers:
            image_embeds = layer(image_embeds)
            layer_number += 1
            if layer_number > len(self.image_transformer.layers) - 3 and layer_number < len(self.image_transformer.layers):
                image_embeds_global.append(image_embeds)
        image_embeds = self.image_transformer.norm(image_embeds)  
        image_embeds_global.append(image_embeds)

        for i in range(len(image_embeds_global)):
            image_embeds = image_embeds_global[i]
            image_embeds = self.image_linears[i](image_embeds)
            avg_image_feats = self.avgpool(image_embeds.transpose(1, 2)).view(image_embeds.size(0), 1, -1)
            cls_feats_image = self.image_pooler(avg_image_feats)
            cls_feats_image = self.normalization_layer(cls_feats_image)
            image_embeds_global[i] = cls_feats_image
            feature, mask = self.local_region_tar[i](image_embeds)
            image_features_local.append(feature)

        return image_embeds_global[0], image_embeds_global[1], image_embeds_global[2], image_features_local[0],image_features_local[1], image_features_local[2]


    def compose_img_text(self, img=None, text_ids=None, text_masks=None, image_token_type_idx=1):  
        text_embeds_global = []
        image_embeds_global = []
        image_features_local = []

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        
        layer_number = 0
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
            layer_number += 1
            if layer_number > len(self.text_transformer.encoder.layer) - 3:
                text_embeds_global.append(text_embeds)
        

        image_embeds = self.image_transformer.patch_embed(img)
        if self.image_transformer.absolute_pos_embed is not None:
            image_embeds = image_embeds + self.image_transformer.absolute_pos_embed
        image_embeds = self.image_transformer.pos_drop(image_embeds)
        layer_number = 0
        for layer in self.image_transformer.layers:
            image_embeds = layer(image_embeds)
            layer_number += 1
            if layer_number > len(self.image_transformer.layers) - 3 and layer_number < len(self.image_transformer.layers):
                image_embeds_global.append(image_embeds)
        image_embeds = self.image_transformer.norm(image_embeds)  
        image_embeds_global.append(image_embeds)


        for i in range(len(text_embeds_global)):
            text_embeds = text_embeds_global[i]
            image_embeds = image_embeds_global[i]

            text_embeds = self.text_linears[i](text_embeds)
            image_embeds = self.image_linears[i](image_embeds)

            image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
            extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

            text_embeds, image_embeds = (
                text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
                image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
            )

            x, y = text_embeds, image_embeds
            for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
                x1 = text_layer(x, y, extend_text_masks, extend_image_masks, True)
                y1 = image_layer(y, x, extend_image_masks, extend_text_masks, True)
                x, y = x1[0], y1[0]

            text_feats, image_feats = x, y
            cls_feats_text = self.text_pooler(x)

            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.image_pooler(avg_image_feats)
            cls_feats_image = self.normalization_layer(cls_feats_image)
            feature, mask = self.local_region_ref[i](image_feats)
            image_features_local.append(feature)
            image_embeds_global[i] = cls_feats_image

        return image_embeds_global[0], image_embeds_global[1], image_embeds_global[2], image_features_local[0],image_features_local[1], image_features_local[2]

    def forward(self, img1, input_ids, input_mask, img2, opt=None, soft_triplet_loss=True):
        return self.compute_loss(img1, input_ids, input_mask, img2, opt)

# class ConCatModule(torch.nn.Module):
#     def __init__(self):
#         super(ConCatModule, self).__init__()
#     def forward(self, x):
#         x = torch.cat(x, 1)
#         return x

# class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
#     """Base class for image and text encoder."""

#     def __init__(self, text_query, image_embed_dim, text_embed_dim, name):
#         super().__init__()
#         # img model
#         img_model = torchvision.models.resnet18(pretrained=True)
#         self.name = name

#         class GlobalAvgPool2d(torch.nn.Module):

#             def forward(self, x):
#                 return F.adaptive_avg_pool2d(x, (1, 1))

#         img_model.avgpool = GlobalAvgPool2d()
#         img_model.fc = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, image_embed_dim))
#         self.img_model = img_model

#         # text model
#         self.text_model = text_model.TextLSTMModel(
#             texts_to_build_vocab = text_query,
#             word_embed_dim = text_embed_dim,
#             lstm_hidden_dim = text_embed_dim)

#     def extract_img_feature(self, imgs):
#         return self.img_model(imgs)

#     def extract_text_feature(self, text_query):
#         return self.text_model(text_query)


# class TIRG(ImgEncoderTextEncoderBase):
#     """The TIRG model.

#     The method is described in
#     Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
#     "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
#     CVPR 2019. arXiv:1812.07119
#     """

#     def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
#         super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)

#         self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
#         self.use_bert = use_bert

#         merged_dim = image_embed_dim + text_embed_dim

#         self.gated_feature_composer = torch.nn.Sequential(
#             ConCatModule(),
#             torch.nn.BatchNorm1d(merged_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(merged_dim, image_embed_dim)
#         )

#         self.res_info_composer = torch.nn.Sequential(
#             ConCatModule(),
#             torch.nn.BatchNorm1d(merged_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(merged_dim, merged_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(merged_dim, image_embed_dim)
#         )

#     def compose_img_text(self, imgs, text_query):
#         img_features = self.extract_img_feature(imgs)
#         text_features = self.extract_text_feature(text_query, self.use_bert)

#         return self.compose_img_text_features(img_features, text_features)

#     def compose_img_text_features(self, img_features, text_features):
#         f1 = self.gated_feature_composer((img_features, text_features))
#         f2 = self.res_info_composer((img_features, text_features))
#         f = F.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]

#         return f