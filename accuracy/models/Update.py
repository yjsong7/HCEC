#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import random
#from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        #self.acc_func, = self.cal_accuracy()
        self.selected_clients = []
        #if args.dataset != 'fashion-mnist':
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        #else:

        #    self.ldr_train = dataset

    def cal_accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train(self, net, local_epoch, lr_local):
        #local_epoch += 1
        net.train()
        # train and update
        #optimizer = torch.optim.SGD(net.parameters(), lr=lr_local, momentum=self.args.momentum)

        epoch_loss = []
        epoch_acc = []

        eta_min = 0.01
        eta_max = 0.015
        lr_local_last = lr_local
        for iter in range(local_epoch): # Ti
            if self.args.CLR == 'True':
            # update local learning rate
                lr_local = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos((iter+1)/local_epoch*math.pi))
                #if iter>0:
                #    lr_local = 0.01 * math.pow(0.25,(iter+1)/local_epoch)

            optimizer = torch.optim.SGD(net.parameters(), lr=lr_local, momentum=self.args.momentum)

            batch_loss = []
            batch_acc = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train): # 对于每份local data，即对于每个分布式节点！
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                if self.args.dataset == 'fashion-mnist':
                    images = images.type(torch.cuda.FloatTensor)
                #print("image type=%s", type(images))
                log_probs = net(images)
                if self.args.dataset == 'fashion-mnist':
                    loss = self.loss_func(log_probs, labels.long())
                else:
                    loss = self.loss_func(log_probs, labels)
                # 准确率
                acc, = self.cal_accuracy(log_probs, labels, topk=(1,))
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
            lr_local_last = lr_local
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)

