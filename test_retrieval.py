# Copyright 2018 Google Inc. All Rights Reserved.
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

# Modified for experiments in https://arxiv.org/abs/2007.00145

"""Evaluates the retrieval model."""
import numpy as np
import torch
from tqdm import tqdm as tqdm
import os
import json

def test(opt, model, tokenizer, testset, filter_categories=False):
    if filter_categories:
        out = []
        for category in testset.categories:
            print("Evaluating on", category)
            cat_out = _test(opt, model, tokenizer, testset, category)
            out += [[category+name, val] for name, val in cat_out]
    else:
        out = _test(opt, model, tokenizer, testset)
    return out


def _test(opt, model, tokenizer, testset, category=None, trnsize=1000, gallery=None):
    """Tests a model over the given testset."""
    model.eval()
    if category is not None:
        test_queries = testset.get_test_queries(category=category)
    else:
        test_queries = testset.get_test_queries()
    
    all_queries = [[] for i in range(6)]
    all_targets = [[] for i in range(6)]
    all_captions = []
    all_target_captions = []

    if test_queries:
        # compute test query features
        imgs = []
        mods = []
        for t in tqdm(test_queries):
            imgs += [testset.get_img(t['source_img_id'])]
            mods += [t['mod']['str']]
            if len(imgs) >= opt.batch_size or t is test_queries[-1]:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()

                encoding = tokenizer(
                    mods,
                    padding="max_length",
                    truncation=True,
                    max_length=opt.max_text_len,
                    return_special_tokens_mask=True,
                    )

                input_ids = torch.tensor(encoding["input_ids"]).cuda()
                input_mask = torch.tensor(encoding["attention_mask"]).cuda()

                f = model.compose_img_text(imgs, input_ids, input_mask)
                for i in range(len(f)):
                    all_queries[i] += [f[i].data.cpu().numpy()]
                imgs = []
                mods = []
        all_target_captions = [tq['target_caption'] for tq in test_queries]

        # compute all image features (within category if applicable)
        all_targets, all_captions = compute_db_features(opt, model, testset, category)

    else:
        # use training queries to approximate training retrieval performance
        # TODO: test that this is doing what it says
        imgs0 = []
        imgs = []
        mods = []
        for i in tqdm(range(trnsize)):
            item = testset[i]
            imgs += [item["source_img_data"]]
            mods += [item['mod']['str']]
            if len(imgs) > opt.batch_size or i == (trnsize-1):
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()

                encoding = tokenizer(
                    mods,
                    padding="max_length",
                    truncation=True,
                    max_length=opt.max_text_len,
                    return_special_tokens_mask=True,
                    )

                input_ids = torch.tensor(encoding["input_ids"]).cuda()
                input_mask = torch.tensor(encoding["attention_mask"]).cuda()

                f = model.compose_img_text(imgs, input_ids, input_mask)
                for i in range(len(all_queries)):
                    all_queries[i] += [f[i].data.cpu().numpy()]
                imgs = []
                mods = []

            imgs0 += [item['target_img_data']]
            if len(imgs0) > opt.batch_size or i == (trnsize-1):
                imgs0 = torch.stack(imgs0).float()
                imgs0 = torch.autograd.Variable(imgs0)
                imgs0 = model.extract_img_feature(imgs0.cuda())
                for i in range(len(all_targets)):
                    all_targets[i] += [imgs0[i].data.cpu().numpy()]
                imgs0 = []
            all_captions += [item['target_caption']]
            all_target_captions += [item['target_caption']]

    for i in range(len(all_queries)):
        all_queries[i] = np.concatenate(all_queries[i])
    for i in range(len(all_targets)):
        all_targets[i] = np.concatenate(all_targets[i])

    nn_result, sorted_sims = nn_and_sims(opt, testset, all_queries, all_targets, test_queries, all_captions, category=category)

    # compute recalls
    out = []
    for k in [1, 5, 10, 50, 100]:
        recall = 0.0
        for i, nns in enumerate(nn_result):
            if all_target_captions[i] in nns[:k]:
                recall += 1
        recall /= len(nn_result)
        out += [('recall_top' + str(k) + '_correct_composition', recall)]

        if opt.dataset == 'mitstates':
            recall = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
                    recall += 1
            recall /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_adj', recall)]

            recall = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
                    recall += 1
            recall /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_noun', recall)]

    return out


def nn_and_sims(opt, testset, all_queries, all_targets, test_queries, all_captions, category=None):
    sims = 0
    for all_queries_item, all_imgs_item in zip(all_queries, all_targets):
        sims = all_queries_item.dot(all_imgs_item.T) + sims
    if test_queries:
        for i, t in enumerate(test_queries):
            source_id = t['source_img_id']
            if opt.VAL_evaluation:
                source_id = all_captions.index(source_id)
            else:
                if category is not None:
                    # get index within category
                    source_id = testset.imgs[source_id]["category"][category]
            sims[i, source_id] = -10e10  # remove query image
            
    nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])]
    nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
    sorted_sims = [np.sort(sims[ii, :])[::-1] for ii in range(sims.shape[0])]

    return nn_result, sorted_sims

def compute_db_features(opt, model, testset, category=None):
    """Compute all image features."""
    all_imgs = [[] for i in range(6)]
    imgs = []
    if opt.VAL_evaluation:
        imset = testset.get_VAL_evaluation_target(category=category)
    else:
        imset = testset.imgs if category is None else testset.img_by_cat[category]
    for image_item in tqdm(imset):
        if opt.VAL_evaluation:
            ind = testset.asin2id[image_item]
        else:
            ind = image_item["image_id"]
        imgs += [testset.get_img(ind)]
        if len(imgs) >= opt.batch_size or image_item is imset[-1]:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float()
            imgs = torch.autograd.Variable(imgs).cuda()
            imgs = model.extract_img_feature(imgs)
            for i in range(len(all_imgs)):
                all_imgs[i] += [imgs[i].data.cpu().numpy()]
            imgs = []
            
    if opt.VAL_evaluation:
        all_captions = [testset.asin2id[image_item] for image_item in imset]
    else:
        all_captions = [image_item["captions"][0] for image_item in imset]
    
    return all_imgs, all_captions


