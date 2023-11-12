#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import torchvision

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

import math
import random
import time

def calculatevec(x,y): # x,y=<class 'collections.OrderedDict'>
    #print(len(x)) # 8 item
    r = copy.deepcopy(x)
    for k in x.keys():
        r[k]=torch.sub(x[k],y[k])
    return r

def calculatemo(x,model):# x,y=<class 'collections.OrderedDict'>
    sum = 0
    #print(len(x))
    if model == 'cnn':
        for k in x.keys(): # 10个
            #print(x[k].numel())
            sum += math.pow(torch.norm(x[k].float()),2)
    else : # resnet50 320个, resnet152 932个
        for i in range(10):
            #_k = random.sample(x.keys(), 1)  # 随机一个字典中的key，第二个参数为限制个数
            _k = random.choice(list(x))  # 选中的key
            #print(type(x[_k])) # tensor
            #print(type(_k[0]))

            while x[_k].numel() < 48000:
                sum += math.pow(torch.norm(x[_k].float()), 2)
    return math.sqrt(sum) # int

def imagenet_prepare():
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    # 使用torchvision.datasets.ImageFolder读取数据集 指定train 和 test文件夹
    trainset = torchvision.datasets.ImageFolder('/home/syj/project/oneIteration/data/IMAGENET/train/', transform=transform)
    #train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=1)
    #train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)

    #trainset = trainset
    #trainloader = train_loader
    train_sampler = None

    return trainset, trainset

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        #dataset_train = torchvision.datasets.ImageFolder('/home/syj/project/federated-learning/data/cifar-10-batches-py/',transform=trans_cifar)
        #dataset_test = torchvision.datasets.ImageFolder('/home/syj/project/federated-learning/data/cifar-10-batches-py/',transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'imagenet':
        trans_imagenet = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train, dataset_test = imagenet_prepare()
        #dataset_test = datasets.CIFAR10('../data/imagenet', train=False, download=True, transform=trans_imagenet)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'imagenet':
        net_glob = torchvision.models.resnet50(num_classes=1000).to(args.device)
    elif args.model == 'resnet50' and args.dataset == 'cifar':
        net_glob = torchvision.models.resnet50(num_classes=10).to(args.device)
    elif args.model == 'resnet50' and args.dataset == 'imagenet':
        net_glob = torchvision.models.resnet50(num_classes=1000).to(args.device)
    elif args.model == 'resnet152' and args.dataset == 'cifar':
        net_glob = torchvision.models.resnet152(num_classes=10).to(args.device)
    elif args.model == 'resnet152' and args.dataset == 'imagenet':
        net_glob = torchvision.models.resnet152(num_classes=1000).to(args.device)
    elif args.model == 'densenet161' and args.dataset == 'cifar':
        net_glob = torchvision.models.densenet161(num_classes=10).to(args.device)
    elif args.model == 'densenet161' and args.dataset == 'imagenet':
        net_glob = torchvision.models.densenet161(num_classes=1000).to(args.device)
    elif args.model == 'densenet201' and args.dataset == 'cifar':
        net_glob = torchvision.models.densenet201(num_classes=10).to(args.device)
    elif args.model == 'densenet201' and args.dataset == 'imagenet':
        net_glob = torchvision.models.densenet201(num_classes=1000).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    #print(net_glob) # 打印网络结构
    net_glob.train() # 训练网络。预热，初始化

    # copy weights
    w_glob = net_glob.state_dict() # <class 'collections.OrderedDict'>

    # training
    loss_train = []
    acc_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # learning rate
    eta_min = 0.0001
    eta_max = 0.01

    Ti_list = []
    #w_round = []
    w_round_last = w_glob
    w_round_current = w_glob

    _filepath = "./save/{}_{}_{}_{}_{}_{}.log".format(args.dataset, args.model, args.epochs, args.local_ep, args.CLR,
                                                     args.ILE)
    mylog = open(_filepath, mode='a+', encoding='utf-8')

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)] # 复制本Ti轮中所有client的权重
    for iter in range(args.epochs): # round
        # adjust local epochs 随便选一个张量，判断tmp值
        tmp = 0
        if iter==0:
            Ti_list.append(args.local_ep)
        else:
            if args.ILE == 'True':
                _tmp = calculatevec(w_round_current, w_round_last) # 平均之后的整个模型参数 <class 'collections.OrderedDict'>
                tmp = calculatemo(_tmp,args.model)/calculatemo(w_round_last,args.model)
                if tmp > 1e-3: # 这个误差值待修改
                    Ti_list.append(Ti_list[iter - 1])
                else:
                    Ti_list.append(2 * Ti_list[iter - 1])
            else:
                Ti_list.append(Ti_list[iter-1])
        print("local epoch=",Ti_list[iter])
        # adjust Ti
        local_epoch = Ti_list[iter]

        calr_round = args.lr
        # adjust lr
        if args.CLR == 'True':
            calr_round = eta_max  # 因为此时为该round中第0个epoch

        w_round_last = w_round_current
        _filepath = "./save/{}_{}_{}_{}_{}_{}.log".format(args.dataset, args.model, args.epochs, args.local_ep,args.CLR,args.ILE)
        mylog = open(_filepath, mode='a+', encoding='utf-8')

        t_start = time.time()
        t_end = 0
        loss_locals = []
        acc_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1) # 设置随机数，选取部分client求平均
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) #选中的client
        for idx in idxs_users: # 每个client
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, acc = local.train(net=copy.deepcopy(net_glob).to(args.device),local_epoch=local_epoch, lr_local=calr_round)
            #print(type(w))
            #print("idx=%d, w.length=%d",idx, len(w))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))


        if iter==0:
            t_end = time.time()
            print("T0 communication interval is",t_end-t_start)
            print('T0 communication interval is {:.3f}'.format(t_end-t_start), file=mylog)
        # update global weights
        w_glob = FedAvg(w_locals) # w_locals的长度是client数，其中每个tensor都是完整模型参数 # <class 'collections.OrderedDict'>
        w_round_current = w_glob  # 记录全局Wi+1 # <class 'collections.OrderedDict'>

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals) # 求本轮的损失值
        acc_avg = sum(acc_locals) / len(acc_locals) # 求本轮的损失值
        print('Round {:3d}, Average loss {:.3f}, Average acc {:.3f}'.format(iter, loss_avg, acc_avg))
        print('Round {:3d}, Average loss {:.3f}, Average acc {:.3f}'.format(iter, loss_avg, acc_avg), file=mylog)
        loss_train.append(loss_avg)
        acc_train.append(acc_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_{}_{}_loss.png'.format(args.dataset, args.model, args.epochs, args.CLR, args.ILE))
    # plot accuracy curve
    plt.figure()
    plt.plot(range(len(acc_train)), acc_train)
    plt.ylabel('train_accuracy')
    plt.savefig('./save/fed_{}_{}_{}_{}_{}_acc.png'.format(args.dataset, args.model, args.epochs, args.CLR, args.ILE))

    print("Ti list",Ti_list)
    print("rounds of training",args.epochs)

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

