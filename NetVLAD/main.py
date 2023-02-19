from __future__ import print_function
from pathlib import Path
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import faiss
import numpy as np
import torchvision.models as models
from skimage.transform import resize

import netvlad
from dataset import get_whole_val_set

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=2, help='Number of threads for each data loader to use')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for testing.')
parser.add_argument('--ckpt', type=str, default='best',
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--root_dir', type=str, default='', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='berlin', 
        help='Dataset to use', choices=['pittsburgh','oxford','darkzurich','nordland','tokyo247','berlin','gardens_point','tb_places'])
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=16, help='Number of NetVlad clusters. Default=16')
parser.add_argument('--save', action='store_true', help='Save NetVLAD Descriptors, Predictions and Cluster Masks')
parser.add_argument('--save_path', type=str, default='', 
        help='Path to save NetVLAD Descriptors, Predictions and Cluster Masks')

def test(eval_set, save_path, epoch=0):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

    cluster_masks = []

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= (opt.num_clusters)
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding, iter_cluster_masks = model.pool(image_encoding)

            cluster_masks.append(iter_cluster_masks)
            
            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')
    
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    _, predictions = faiss_index.search(qFeat, max(n_values))

    # Save NetVLAD Descriptors, Predictions and Cluster Masks
    cluster_masks = torch.cat(cluster_masks, dim=0)
    cluster_masks = cluster_masks.detach().cpu().numpy()
    cluster_masks = np.reshape(cluster_masks, (cluster_masks.shape[0],30,40))
    cluster_masks = resize(cluster_masks, (cluster_masks.shape[0],480,640), order=0, preserve_range=True)

    if opt.save:
      np.save(join(save_path, 'dbFeat.npy'), dbFeat)
      np.save(join(save_path, 'qFeat.npy'), qFeat)
      np.save(join(save_path, 'netvlad_preds.npy'), predictions)
      np.save(join(save_path, 'db_cluster_masks.npy'), cluster_masks[:eval_set.dbStruct.numDb])
      np.save(join(save_path, 'q_cluster_masks.npy'), cluster_masks[eval_set.dbStruct.numDb:])
    
    # for each query get those within threshold distance
    gt = eval_set.getPositives()
    
    correct_at_n = np.zeros(len(n_values))
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break

    recall_at_n = correct_at_n / eval_set.dbStruct.numQ

    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

    return recalls

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

if __name__ == "__main__":
    opt = parser.parse_args()

    restore_var = ['num_clusters', 'pooling', 'seed']
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these 
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)

    save_path = Path(opt.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')
    
    whole_test_set = get_whole_val_set(opt.root_dir, opt.dataset.lower())
    print('===> Evaluating')
    print('====> Query count:', whole_test_set.dbStruct.numQ)

    print('===> Building model')

    pretrained = True
    if opt.arch.lower() == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)

    if opt.pooling.lower() == 'netvlad':
        net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
        model.add_module('pool', net_vlad)
    elif opt.pooling.lower() == 'max':
        global_pool = nn.AdaptiveMaxPool2d((1,1))
        model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    elif opt.pooling.lower() == 'avg':
        global_pool = nn.AdaptiveAvgPool2d((1,1))
        model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    else:
        raise ValueError('Unknown pooling type: ' + opt.pooling)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if opt.ckpt.lower() == 'latest':
        resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
    elif opt.ckpt.lower() == 'best':
        resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        best_metric = checkpoint['best_score']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_ckpt))

    print('===> Running evaluation step')
    epoch = 1
    recalls = test(whole_test_set, opt.save_path, epoch)