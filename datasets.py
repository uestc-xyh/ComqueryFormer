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
"""Great things."""

"""Provides data for training and testing."""
import numpy as np
import PIL
import skimage
import torch
import json
import torch.utils.data
import torchvision
import warnings
import random
import os

class BaseDataset(torch.utils.data.Dataset):
    """Base class for a dataset."""

    def __init__(self):
        super(BaseDataset, self).__init__()
        self.imgs = []
        self.test_queries = []

    def get_loader(self,
                   datasampler,
                   batch_size=32,
                   shuffle=False,
                   drop_last=False,
                   num_workers=0):
        '''
        return torch.utils.data.DataLoader(
            self,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)
        '''
        return torch.utils.data.DataLoader(
            self,
            sampler=datasampler,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def get_test_queries(self):
        return self.test_queries

    def get_all_texts(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.generate_random_query_target()

    def generate_random_query_target(self):
        raise NotImplementedError

    def get_img(self, idx, raw_img=False):
        raise NotImplementedError

class Fashion200k(BaseDataset):
    """Fashion200k dataset."""
    def __init__(self, path, split='train', transform=None):
        super(Fashion200k, self).__init__()

        self.split = split
        self.transform = transform
        self.img_path = path + '/'

        # get label files for the split
        label_path = path + '/labels/'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if split in f]

        # read image info from label files
        self.imgs = []

        def caption_post_process(s):
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                '&', 'andmark').replace('*', 'starmark')

        for filename in label_files:
            print('read ' + filename)
            with open(label_path + '/' + filename) as f:
                lines = f.readlines()

            for line in lines:
                line = line.split('	')
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])],
                    'split': split,
                    'modifiable': False
                }
                self.imgs += [img]
        print('Fashion200k:', len(self.imgs), 'images')

        # generate query for training or testing
        if split == 'train':
            self.caption_index_init_()
        else:
            self.generate_test_queries_()

    def get_different_word(self, source_caption, target_caption):
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str

    def generate_test_queries_(self):
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        with open(self.img_path + '/test_queries.txt') as f:
            lines = f.readlines()
        self.test_queries = []
        for line in lines:
            source_file, target_file = line.split()
            try:
                idx = file2imgid[source_file]
            except:
                continue

            try:
                target_idx = file2imgid[target_file]
            except:
                continue

            source_caption = self.imgs[idx]['captions'][0]
            target_caption = self.imgs[target_idx]['captions'][0]
            source_word, target_word, mod_str = self.get_different_word(
                source_caption, target_caption)
            self.test_queries += [{
                'source_img_id': idx,
                'source_caption': source_caption,
                'target_caption': target_caption,
                'mod': {
                    'str': mod_str
                }
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print(len(caption2imgids), 'unique cations')

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
        for img in self.imgs:
            img['modifiable'] = False
            img['parent_captions'] = []
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.imgs[imgid]['modifiable'] = True
                        self.imgs[imgid]['parent_captions'] += [p]
        num_modifiable_imgs = 0
        for img in self.imgs:
            if img['modifiable']:
                num_modifiable_imgs += 1
        print('Modifiable images', num_modifiable_imgs)

    def caption_index_sample_(self, idx):
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # find random target image (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
        return texts

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(
            idx)

        out = {}
        return{
        'source_img_id' : idx,
        'source_img_data' : self.get_img(idx),
        'source_caption' : self.imgs[idx]['captions'][0],
        'target_img_id' : target_idx,
        'target_img_data' : self.get_img(target_idx),
        'target_caption' : self.imgs[target_idx]['captions'][0],
        'mod' : {
            'str': mod_str
            }
        }

    def get_img(self, idx, raw_img=False):
        img_path = self.img_path + self.imgs[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img


CATEGORIES = ["dress", "shirt", "toptee"]

class FashionIQ(BaseDataset):
    def __init__(self, path, split='train', transform=None, batch_size=None):
        super(FashionIQ, self).__init__()
        self.categories = CATEGORIES

        self.split = split
        self.transform = transform
        self.img_path = path + '/'

        failures = []

        data = {
            'image_splits': {},
            'captions': {}
        }

        for data_type in data:
            for datafile in os.listdir(path + '/' + data_type):
                if split in datafile:
                    if '.pkl' in datafile:
                        continue
                    data[data_type][datafile] = json.load(open(path + '/' + data_type + '/' + datafile, 'rb'))

        split_labels = sorted(list(data["image_splits"].keys()))
        global_imgs = []
        img_by_cat = {cat: [] for cat in CATEGORIES}
        self.asin2id = {}
        for splabel in split_labels:
            for asin in data['image_splits'][splabel]:
                category = splabel.split(".")[1]
                file_path = path + '/image_data/' + category + '/' + asin + '.jpg'
                if os.path.exists(file_path) or split == "test":
                    global_id = len(global_imgs)
                    category_id = len(img_by_cat[category])
                    entry = [{
                        'asin': asin,
                        'file_path': file_path,
                        'captions': [global_id],
                        "image_id": global_id,
                        "category": {category: category_id}
                    }]
                    if asin in self.asin2id:
                        # handle duplicates
                        oldglobal = self.asin2id[asin]
                        subentry = global_imgs[oldglobal]
                        assert category not in subentry["category"], \
                            "{} duplicated in {}".format(asin, category)

                        # update entry to include additional category and id
                        subentry["category"][category] = category_id
                        img_by_cat[category] += [subentry]
                    else:
                        # just add the entry
                        global_imgs += entry
                        img_by_cat[category] += entry
                        self.asin2id[asin] = global_id
                else:
                    failures.append(asin)

        assert len(global_imgs) > 0, "no data found"

        queries = []
        captions = sorted(list(data["captions"].keys()))
        for cap in captions:
            for query in data['captions'][cap]: # query has "target", "captions" and ""candidate"" key
                if split != "test" and (query['candidate'] in failures
                                        or query.get('target') in failures):
                    continue
                query['source_id'] = self.asin2id[query['candidate']]
                query['source_name'] = query['candidate']
                query["category"] = cap.split(".")[1]
                if split != "test":
                    query['target_id'] = self.asin2id[query['target']]
                    query['target_name'] = query['target']
                    tarcat = global_imgs[query['target_id']]["category"]
                    if query["category"] not in tarcat:
                        print("WARNING: a {} found with a target in {}".format(
                            query["category"], tarcat
                        ))
                soucat = global_imgs[query['source_id']]["category"]
                assert query["category"] in soucat
                queries += [query]

        self.data = data
        self.imgs = global_imgs
        self.img_by_cat = img_by_cat
        self.queries = queries

        self.img_by_cat_VAL_evaluation = {cat: [] for cat in CATEGORIES}
        if split == "val":
            for query in queries:
                query_item_one = {
                  'source_img_id': query['source_id'],
                  'target_img_id': query['target_id'],
                  'target_caption': query['target_id'],
                  'mod': {'str': query['captions'][0] + ' and ' +
                                 query['captions'][1]},
                  "category": query["category"]
                  }
                query_item_two = {
                  'source_img_id': query['source_id'],
                  'target_img_id': query['target_id'],
                  'target_caption': query['target_id'],
                  'mod': {'str': query['captions'][1] + ' and ' +
                                 query['captions'][0]},
                  "category": query["category"]
                }
                self.test_queries.append(query_item_one)
                self.test_queries.append(query_item_two)
                self.img_by_cat_VAL_evaluation[query["category"]].append(query['source_name'])
                self.img_by_cat_VAL_evaluation[query["category"]].append(query['target_name'])

            for cat in CATEGORIES:
                 self.img_by_cat_VAL_evaluation[cat] = list(set(self.img_by_cat_VAL_evaluation[cat]))

        if split == "test":
            for query in queries:
                query_item = {
                  'source_img_id': query['source_id'],
                  'mod': {'str': query['captions'][0] + ' and ' +
                                 query['captions'][1]},
                  "category": query["category"]
              }
                self.test_queries.append(query_item)

        self.id2asin = {val: key for key, val in self.asin2id.items()}
        self.current_category = CATEGORIES[0]

    def get_all_texts(self):
        texts = [' and ']
        for query in self.queries:
            texts += query['captions'][0] + query['captions'][1]
        return texts

    def __len__(self):
        return len(self.queries)*2

    def get_loader(self,
                   batch_size,
                   shuffle=False,
                   drop_last=False,
                   num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def __getitem__(self, idx):
        safe_idx = idx // 2
        reverse = (idx % 2 == 1)

        query = self.queries[safe_idx]

        if not reverse:
            mod_str = query['captions'][0] + ' and ' + query['captions'][1]
        else:
            mod_str = query['captions'][1] + ' and ' + query['captions'][0]

        return {
          'source_img_id': query['source_id'],
          'source_img_data': self.get_img(query['source_id']),
          'target_img_id': query['target_id'],
          'target_caption': query['target_id'],
          'target_img_data': self.get_img(query['target_id']),
          'mod': {'str': mod_str}
        }

    def get_img(self, idx, raw_img=False):
        """Retrieve image by global index."""
        img_path = self.imgs[idx]['file_path']
        try:
            with open(img_path, 'rb') as f:
                img = PIL.Image.open(f)
                img = img.convert('RGB')
        except EnvironmentError as ee:
            print("WARNING: EnvironmentError, defaulting to image 0", ee)
            img = self.get_img(0, raw_img=True)
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img

    def get_test_queries(self, category=None):
        if category is not None:
            return [que for que in self.test_queries if que["category"] == category]
        return self.test_queries

    def get_VAL_evaluation_target(self, category=None):
        test_sets = self.img_by_cat_VAL_evaluation[category]
        return test_sets