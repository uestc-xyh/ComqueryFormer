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

"""Main method to train the model."""
# !/usr/bin/python
import os
import random
import argparse
import gc
import time

import test_retrieval

from tensorboardX import SummaryWriter
from torch.autograd import Variable
import datasets

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm
from copy import deepcopy
from datetime import datetime
import numpy as np

import img_text_composition_models
from transformers import BertTokenizer
import test_retrieval
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(3)

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--model', type=str, default='tirg|comqueryformer')
    parser.add_argument('--image_embed_dim', type=int, default=512)
    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument('--use_complete_text_query', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--category_to_train', type=str, default='all')
    parser.add_argument('--loss', type=str, default='soft_triplet|batch_based_classification')
    parser.add_argument('--loader_num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='../logs/')
    parser.add_argument('--test_only', type=bool, default=False)
    parser.add_argument('--model_checkpoint', type=str, default='')
    parser.add_argument('--VAL_evaluation', type=bool, default=False)

    # Image setting
    parser.add_argument("--image_size", default=224, type=int, help='Number of image_size.')
    parser.add_argument("--patch_size", default=32, type=int, help='Number of patch_size.')

    # Text Setting
    parser.add_argument("--max_text_len", default=40, type=int, help='Number of max_text_len.')
    parser.add_argument("--vocab_size", default=30522, type=int, help='Number of vocab_size.')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased')

    # Transformer Setting
    parser.add_argument("--input_image_embed_size", default=[384, 768, 768], type=list, help='Number of input_image_embed_size.')
    parser.add_argument("--input_text_embed_size", default=[768, 768, 768], type=list, help='Number of input_text_embed_size.')

    parser.add_argument('--vit', type=str, default='swin_tiny_patch4_window7_224')
    parser.add_argument("--hidden_size", default=768, type=int, help='Number of hidden_size.')
    parser.add_argument("--num_layer", default=4, type=int, help='Number of num_layer.')
    parser.add_argument("--num_heads", default=8, type=int, help='Number of num_heads.')
    parser.add_argument("--mlp_ratio", default=4, type=int, help='Number of mlp_ratio.')
    parser.add_argument("--drop_rate", default=0.1, type=int, help='Number of drop_rate.')

    # others
    parser.add_argument("--epochs", default=100, type=int, help='Number of epochs.')
    parser.add_argument("--num_clusters", default=8, type=int, help='Number of num_clusters.')
    args = parser.parse_args()
    return args


def load_dataset(opt):
    """Loads the input datasets."""
    print('Reading dataset ', opt.dataset)
    if opt.dataset == 'fashion200k':
        trainset = datasets.Fashion200k(
            path=opt.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.Fashion200k(
            path=opt.dataset_path,
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
    elif opt.dataset == 'fashionIQ':
        trainset = datasets.FashionIQ(
            path=opt.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(size=224, scale=(0.75, 1.33)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ])
            )
        testset = datasets.FashionIQ(
            path=opt.dataset_path,
            split='val',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(size=224, scale=(0.75, 1.33)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ])
            )
    else:
        print('Invalid dataset', opt.dataset)
        sys.exit()

    print('trainset size:', len(trainset))
    print('testset size:', len(testset))
    return trainset, testset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True

def create_model_and_optimizer(opt, texts):
    """Builds the model and related optimizer."""
    print("Creating model and optimizer for", opt.model)
    if opt.model == 'tirg':
        model = img_text_composition_models.TIRG(texts,
                                                 image_embed_dim=opt.image_embed_dim,
                                                 text_embed_dim=text_embed_dim,
                                                 name= opt.model)
    elif opt.model == 'comqueryformer':
        model = img_text_composition_models.ComqueryFormer(opt)
        # Using the bert tokenizer
        tokenizer = BertTokenizer.from_pretrained(opt.tokenizer, do_lower_case= "uncased" in opt.tokenizer)

    model = model.cuda()

    return model, tokenizer

def run_eval(opt, logger, trainset, testset, model, tokenizer, it, eval_on_val_method=False, eval_on_test=False):
    tests = []
    for name, dataset in [('train', trainset)]:
        categ = opt.dataset == "fashioniq" and name == 'test'
        t = test_retrieval.test(opt, model, tokenizer, dataset, filter_categories=categ)
        tests += [(name + ' ' + metric_name, metric_value)
                for metric_name, metric_value in t]
    for metric_name, metric_value in tests:
        logger.add_scalar(metric_name, metric_value, it)
        print('    ', metric_name, round(metric_value, 4))

    tests = []
    for name, dataset in [('test', testset)]:
        categ = opt.dataset == "fashionIQ" and name == 'test'
        t = test_retrieval.test(opt, model, tokenizer, dataset, filter_categories=categ)
        tests += [(name + ' ' + metric_name, metric_value)
                for metric_name, metric_value in t]
    for metric_name, metric_value in tests:
        logger.add_scalar(metric_name, metric_value, it)
        print('    ', metric_name, round(metric_value, 4))
    if opt.dataset == "fashionIQ":
        scores = [metric for name, metric in tests if
                "test" in name and ("top10_" in name or "top50_" in name)]
        fiq_score = np.mean(scores)
        logger.add_scalar("fiq_score", fiq_score, it)
        print('    ', 'fiq_score', round(fiq_score, 4))

def train_loop(opt, loss_weights, logger, trainset, testset, model, optimizer, scheduler, tokenizer):
    """Function for train loop"""
    print('Begin training')
    
    losses_tracking = {}
    it = 0
    epoch = 0  
    tic = time.time()

    while epoch < opt.epochs:
        epoch += 1
        # show/log stats
        print('It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic,
                                                              4), opt.comment)

        tic = time.time()


        for loss_name in losses_tracking:
            avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
            print('    Loss', loss_name, round(avg_loss, 4))
            logger.add_scalar(loss_name, avg_loss, it)
        print("learning_rate is : ", optimizer.param_groups[0]['lr'])
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)


        if epoch % 1 == 0:
            gc.collect()


        # test
        if (epoch % 5 == 0):
            run_eval(opt, logger, trainset, testset, model, tokenizer, it)
            # save checkpoint
            torch.save({
                'it': it,
                'opt': opt,
                'model_state_dict': model.state_dict(),},
                logger.file_writer.get_logdir() + '/latest_checkpoint.pth')

        # run training for 1 epoch
        model.train()
        trainloader = torch.utils.data.DataLoader(
                                trainset,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=opt.loader_num_workers,
                                drop_last=True,
                                collate_fn=lambda i: i)

        def training_1_iter(data):
            # compute loss
            if opt.loss not in ['soft_triplet', 'batch_based_classification']:
                print('Invalid loss function', opt.loss)
                sys.exit()
            losses = []
            if_soft_triplet = True if opt.loss == 'soft_triplet' else False

            assert type(data) is list
            img1 = np.stack([d['source_img_data'] for d in data])
            img1 = torch.from_numpy(img1).float()
            img1 = torch.autograd.Variable(img1).cuda()

            img2 = np.stack([d['target_img_data'] for d in data])
            img2 = torch.from_numpy(img2).float()
            img2 = torch.autograd.Variable(img2).cuda()
            
            if opt.use_complete_text_query:
                if opt.dataset == 'mitstates':
                    supp_text = [str(d['noun']) for d in data]
                    mods = [str(d['mod']['str']) for d in data]
                    # text_query here means complete_text_query
                    text_query = [adj + " " + noun for adj, noun in zip(mods, supp_text)]
                else:
                    text_query = [str(d['target_caption']) for d in data]
            else:
                text_query = [str(d['mod']['str']) for d in data]

            encoding = tokenizer(
                text_query,
                padding="max_length",
                truncation=True,
                max_length=opt.max_text_len,
                return_special_tokens_mask=True,
                )

            input_ids = torch.tensor(encoding["input_ids"]).cuda()
            input_mask = torch.tensor(encoding["attention_mask"]).cuda()

            loss_value = model(img1, input_ids, input_mask, img2, opt=opt, soft_triplet_loss=if_soft_triplet)
            loss_name = opt.loss

            loss_value = loss_value[0]
            loss_value = loss_value.mean()  

            losses += [(loss_name, loss_weights[0], loss_value)]

            total_loss = sum([
                loss_weight * loss_value
                for loss_name, loss_weight, loss_value in losses
            ])
            assert not torch.isnan(total_loss)
            losses += [('total training loss', None, total_loss.item())]

            # track losses
            for loss_name, loss_weight, loss_value in losses:
                if loss_name not in losses_tracking:
                    losses_tracking[loss_name] = []
                losses_tracking[loss_name].append(float(loss_value))

            torch.autograd.set_detect_anomaly(True)

            # gradient descendt
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        count = 0
        for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):

            it += 1
            training_1_iter(data)

    print('Finished training')

def get_optimizer(model, opt):
    lr = opt.learning_rate
    feature_encoder_name = ['image_transformer', 'text_transformer']

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in feature_encoder_name)
            ],
            "lr": lr*5, # (le-5)*5
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in feature_encoder_name)
            ],
            "lr": lr, # (le-5)
        },
    ]
    return optimizer_grouped_parameters

def main():

    set_seed(666)
    opt = parse_opt()
    opt.use_complete_text_query = False
    print('Arguments:')
    for k in opt.__dict__.keys():
        print('    ', k, ':', str(opt.__dict__[k]))

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    loss_weights = [1.0, 0.1, 0.1, 0.01]
    logdir = os.path.join(opt.log_dir, current_time + '_' + opt.comment)

    print("logdir is : ", logdir)
    logger = SummaryWriter(logdir)

    print('Log files saved to', logger.file_writer.get_logdir())
    for k in opt.__dict__.keys():
        logger.add_text(k, str(opt.__dict__[k]))

    trainset, testset = load_dataset(opt)
    opt.t_total = len(trainset)/opt.batch_size * opt.epochs

    model, tokenizer = create_model_and_optimizer(opt, [t for t in trainset.get_all_texts()])

    from transformers import AdamW, get_linear_schedule_with_warmup
    model_parameters = get_optimizer(model, opt)
    optimizer = AdamW(model_parameters, lr=opt.learning_rate)
    warmup = 0.1 
    scheduler = get_linear_schedule_with_warmup(optimizer, opt.t_total * warmup, opt.t_total)
    
    if opt.test_only:
        print('Doing test only')
        # opt.model_checkpoint = ''
        checkpoint = torch.load(opt.model_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        it = checkpoint['it']
        model.eval()
        tests = []
        it = 0
        run_eval(opt, logger, trainset, testset, model, tokenizer, it)

        return 0

    train_loop(opt, loss_weights, logger, trainset, testset, model, optimizer, scheduler, tokenizer)
    logger.close()


if __name__ == '__main__':
    main()
