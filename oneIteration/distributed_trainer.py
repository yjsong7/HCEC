# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import torch
import numpy as np
import argparse, os
import settings
import utils
import logging
import distributed_optimizer

from dl_trainer import DLTrainer, _support_datasets, _support_dnns
if settings.ORIGINAL_HOROVOD:
    import horovod.torch as hvd
else:
    import distributed_optimizer as hvd
    os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
    os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
    os.environ['HOROVOD_MPI_THREADS_DISABLE'] = '1'

from compression import compressors
from profiling import benchmark
from mpi4py import MPI
comm = MPI.COMM_WORLD
from tensorboardX import SummaryWriter
from settings import logger, formatter



# def stwfbp(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, threshold, gradient_path=None):
#     rank = hvd.rank() #GPU在环中的索引
#     torch.cuda.set_device(rank%nwpernode)
#     if rank != 0:
#         pretrain = None
#     writer = SummaryWriter('./tb_log')
#     # ngpus = 1 要不要改？ 后面有分叉
#     trainer = DLTrainer(rank, nworkers, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix='allreduce', pretrain=pretrain, num_steps=num_steps, tb_writer=writer)
#     # 初始化DLTrainer参数，和一些默认配置
#     init_epoch = torch.ones(1) * trainer.get_train_epoch() # trainer类中包括了训练过程中的一些属性
#     init_iter = torch.ones(1) * trainer.get_train_iter()
#     trainer.set_train_epoch(int(hvd.broadcast(init_epoch, root_rank=0)[0]))
#     trainer.set_train_iter(int(hvd.broadcast(init_iter, root_rank=0)[0]))
#     is_sparse = density < 1
#     if not is_sparse:
#         compressor = None
#
#     # 在layer-wise的基础上，实现梯度合并
#     if settings.ADAPTIVE_MERGE:
#         seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer) # 从训练benchmark中获取3个值
#         layerwise_times = comm.bcast(layerwise_times, root=0)
#         if rank == 0:
#             #logger.info('layerwise seq_layernames: %s', list(seq_layernames))
#             #logger.info('layerwise backward times: %s', list(layerwise_times))
#             logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
#         logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
#         logger.info('Model size: %d', np.sum(layerwise_sizes))
#     else:
#         seq_layernames, layerwise_times, layerwise_sizes = None, None, None
#
#     # benchmark的计算已结束，获得了 每层的key名字[layername]，每层BP耗时layerwise_times, 每层参数（W或b）的大小layerwise_sizes
#     norm_clip = None
#     if dnn == 'lstm':
#         norm_clip = 0.25
#     elif dnn == 'lstman4':
#         norm_clip = 400
#
#     # 此处的hvd，是distributed_optimizer.py下修改后的hvd哦！！
#     # 初始化了optimizer对象，执行_init_中的代码（产生合并方案、注册hook等）
#     if settings.ORIGINAL_HOROVOD: # 确定optimizer（比如是选择SGD 还是Adam 还是自定义的）
#         optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor])
#     else:
#         optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor], seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path)
#
#     hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0) # 从rank开始，广播参数和参数dict
#     trainer.update_optimizer(optimizer) # self.optimizer = optimizer (自定义的）
#
#     iters_per_epoch = trainer.get_num_of_training_samples() // (nworkers * batch_size * nsteps_update)
#
#     times = []
#     logger.info('max_epochs: %d', max_epochs)
#     display = 10 if iters_per_epoch > 10 else iters_per_epoch-1
#     each_epoch = []
#     each_display = []
#     mylog = open(settings.LOGGER_PATH, mode='w', encoding='utf-8')
#     for epoch in range(max_epochs):
#         hidden = None
#         if dnn == 'lstm':
#             hidden = trainer.net.init_hidden()
#         for i in range(iters_per_epoch): # 每个epoch里的每次迭代
#             s = time.time() # 本次迭代开始时间
#             optimizer.zero_grad() # 在开始本轮迭代前，先清空梯度，否则会累加
#             for j in range(nsteps_update): # 梯度累加，多少steps后才更新梯度
#                 if j < nsteps_update - 1 and nsteps_update > 1:
#                     optimizer.local = True # 累加，还不聚合
#                 else:
#                     optimizer.local = False # 需要聚合、更新
#                 if dnn == 'lstm':
#                     _, hidden = trainer.train(1, hidden=hidden)
#                 else:
#                     trainer.train(1) # 训练 完成1次迭代 中的前向+反向传播
#             if dnn == 'lstm':
#                 optimizer.synchronize() # dist_optimizer中的sychronize。获取了每个合并小组的allreduce时间列表
#                 # 每个worker的梯度由原生optimizer计算，梯度同步时才由DistributedOptimizer计算
#                 torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
#             elif dnn == 'lstman4':
#                 optimizer.synchronize()
#                 torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
#             trainer.update_model() # 在此处计算、更新。调用了dist_optimizer.step() -> synchronize()
#             times.append(time.time()-s) # 本轮迭代的全部耗时
#
#             if i % display == 0 and i > 0: # 每display次迭代，打印一次
#                 time_per_iter = np.mean(times)
#                 each_display.append(np.sum(times))
#                 print("Time per iteration including communication: %f, Speed: %f images/s", time_per_iter, batch_size * nsteps_update / time_per_iter, file=mylog)
#                 times = []
#
#         t = np.sum(each_display)
#         each_epoch.append(t)
#         print("Time of %d epoch: %f", epoch, t, file=mylog)
#         each_display = []
#         if not settings.ORIGINAL_HOROVOD:
#             optimizer.train_epoch += 1
#     print("Time of all epochs: %f", np.sum(each_epoch), file=mylog)
#     mylog.close()

def stwfbp(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, threshold, gradient_path=None):
    rank = hvd.rank() #GPU在环中的索引
    torch.cuda.set_device(rank%nwpernode)
    if rank != 0:
        pretrain = None
    writer = SummaryWriter('./tb_log')
    # ngpus = 1 每个机器上的GPU数
    trainer = DLTrainer(rank, nworkers, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix='allreduce', pretrain=pretrain, num_steps=num_steps, tb_writer=writer)
    #trainer = DLTrainer(rank, nworkers, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=2, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix='allreduce', pretrain=pretrain, num_steps=num_steps, tb_writer=writer)
    # 初始化DLTrainer参数，和一些默认配置
    init_epoch = torch.ones(1) * trainer.get_train_epoch() # trainer类中包括了训练过程中的一些属性
    init_iter = torch.ones(1) * trainer.get_train_iter()
    trainer.set_train_epoch(int(hvd.broadcast(init_epoch, root_rank=0)[0]))
    trainer.set_train_iter(int(hvd.broadcast(init_iter, root_rank=0)[0]))
    is_sparse = density < 1
    if not is_sparse:
        compressor = None

    # 在layer-wise的基础上，实现梯度合并
    if settings.ADAPTIVE_MERGE:
        seq_layernames, layerwise_times, layerwise_sizes, tftb = benchmark(trainer) # 从训练benchmark中获取3个值
        layerwise_times = comm.bcast(layerwise_times, root=0)
        if rank == 0:
            #logger.info('layerwise seq_layernames: %s', list(seq_layernames))
            #logger.info('layerwise backward times: %s', list(layerwise_times))
            logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
        logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
        logger.info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None

    # benchmark的计算已结束，获得了 每层的key名字[layername]，每层BP耗时layerwise_times, 每层参数（W或b）的大小layerwise_sizes
    norm_clip = None
    if dnn == 'lstm':
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    # 此处的hvd，是distributed_optimizer.py下修改后的hvd哦！！
    # 初始化了optimizer对象，执行_init_中的代码（产生合并方案、注册hook等）
    optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters(),
                                         compression=compressors[compressor], seq_layernames=seq_layernames,
                                         layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold,
                                         writer=writer, gradient_path=gradient_path)

    #optimizer._benchmark_communication()# 测量通信速率用，修改了169行，恢复需要用前面注释的那段代码
    #hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0) # 从rank开始，广播参数和参数dict
    #trainer.update_optimizer(optimizer) # self.optimizer = optimizer (自定义的）


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AllReduce trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=1, help='Just for experiments, and it cannot be used in production')
    parser.add_argument('--nwpernode', type=int, default=1, help='Number of workers per node')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=_support_datasets, help='Specify the dataset for training')
    #parser.add_argument('--dnn', type=str, default='resnet20', choices=_support_dnns, help='Specify the neural network for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=_support_dnns, help='Specify the neural network for training')
    #parser.add_argument('--dnn', type=str, default='googlenet', choices=_support_dnns, help='Specify the neural network for training')
    #parser.add_argument('--dnn', type=str, default='lstm', choices=_support_dnns, help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--saved-dir', type=str, default='.', help='Specify the saved weights or gradients root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--max-epochs', type=int, default=settings.MAX_EPOCHS, help='Default maximum epochs to train')
    parser.add_argument('--pretrain', type=str, default=None, help='Specify the pretrain path')
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--compressor', type=str, default='sigmathresallgather', choices=compressors.keys(), help='Specify the compressors if \'compression\' is open')
    parser.add_argument('--density', type=float, default=1, help='Density for sparsification')
    parser.add_argument('--threshold', type=int, default=0, help='Specify the threshold for gradient merging')
    #parser.add_argument('--printDir',type=str,default='./print/print-default.log')
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update

    prefix = settings.PREFIX
    if args.density < 1:
        prefix = 'comp-' + args.compressor + '-' + prefix
    logdir = 'allreduce-%s-thres-%dkbytes/%s-n%d-bs%d-lr%.4f-ns%d-ds%s' % (prefix, args.threshold/1024, args.dnn, args.nworkers, batch_size, args.lr, args.nsteps_update, str(args.density))
    relative_path = './logs/%s'%logdir
    gradient_relative_path = None
    utils.create_path(relative_path)
    rank = 0

    if args.nworkers > 1: # 若多机
        hvd.init() # 初始化horovod环境，启动相关线程和MPI
        rank = hvd.rank() # 各进程获取rank
        # For example, you have 4 nodes and 4 GPUs each node, so you spawn 16 workers. Every worker will have a rank [0, 15], and every worker will have a local_rank [0, 3].
    else:
        hvd.init()
        rank = 0
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = None #SummaryWriter(tb_runs)
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.info('Configurations: %s', args)
    stwfbp(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.threshold, gradient_relative_path)
