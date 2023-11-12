from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import time
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import models.lstm as lstmpy
from tensorboardX import SummaryWriter
from horovod.torch.mpi_ops import rank
from settings import logger
class Profiling(object):
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            raise ValueError("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self._parameter_names = {v: k for k, v
                                in model.named_parameters()}
        self._seq_keys = [k for k, v in model.named_parameters()]
        self._backward_seq_keys = []
        self._backward_key_sizes = []
        self._grad_accs = []
        self._handles = {} # 字典，<name, [handle_time]> 是吗？ handle_time有几个，这个层就被算过了几次
        self.hook_done = False
        self._start = time.time()
        self._register_hooks()
        self._is_profiling = False

        #print(k for k in model.named_parameters())

    def _register_hooks(self):
        for name, p in self.model.named_parameters(): # [name, param]
            p.register_hook(self._make_hook(name, p)) # register_hook(函数)

    def _make_hook(self, name, p): # 这个函数是在BP过程中额外实现的内容（不然BP结束，中间结果被释放掉了）
        def hook(*ignore):
            if not self._is_profiling:
                return
            name = self._parameter_names.get(p)
            if len(self._backward_seq_keys) != len(self._seq_keys):
                self._backward_seq_keys.append(name)
                self._backward_key_sizes.append(p.numel())
            if name not in self._handles: # 该层还没被处理过，初始化
                self._handles[name] = []
            torch.cuda.synchronize() # 等待该gpu上完成
            ct = self._timestamp(name)
            self._handles[name].append(ct - self._start) # 存储了每层参数从开始到注册完成的耗时
        return hook

    def reset_start(self):
        self._start = time.time()

    def reset(self):
        self._start = time.time()
        self._handles.clear()

    def stop(self):
        self._is_profiling = False

    def start(self):
        self._is_profiling = True
        self._start = time.time()

    def get_backward_seq_keys(self):
        return self._backward_seq_keys

    def get_backward_key_sizes(self):
        return self._backward_key_sizes

    def get_layerwise_times(self):
        num_trials = len(self._handles[self._seq_keys[0]]) # 这层被处理了几次
        layerwise_times_multipletest = []
        totals = []
        for j in range(num_trials):
            s = 0
            total = 0.0
            layerwise_times = [] # from the last layer to the first layer
            #for i, k in enumerate(self._seq_keys[::-1]):
            for i, k in enumerate(self._backward_seq_keys):
                t = self._handles[k][j] # k这层被处理的第j次  的时间
                #print('name: ', k, ' diff: ', t-s)
                layerwise_times.append(t-s)
                total += (t-s)
                s = total
            layerwise_times_multipletest.append(layerwise_times) # 单独这一层的耗时
            totals.append(total)
        array = np.array(layerwise_times_multipletest)
        layerwise_times = np.mean(array, axis=0) # 输出矩阵是1行，求的是每一列的平均值
        return layerwise_times, np.mean(totals) # 1行，1个平均值

    def _timestamp(self, name):
        return time.time()


def benchmark(trainer):
    # Benchmark to achieve the backward time per layer
    p = Profiling(trainer.net) # p=类[网络模型的相关参数]
    # Warmup
    input_shape, output_shape = trainer.get_data_shape()
    warmup = 5 # warmup should be 0 on some GPUs (e.g., P102-100)
    iteration = 50
    start_time = time.time()
    for i in range(iteration+warmup): # 对于第i轮迭代
        data = trainer.data_iter() # 读数据

        if trainer.dataset == 'an4':
            inputs, labels_cpu, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        else:
            inputs, labels_cpu = data
        if trainer.is_cuda: # 只要ngpus>0（有gpu），此值为true
            # .cuda() 是为了把模型放在gpu上训练。数据类型cpu转gpu：data.cuda ,gpu转cpu： data.cpu
            # （数据拷贝）
            if trainer.dnn == 'lstm' :
                inputs = Variable(inputs.transpose(0, 1).contiguous()).cuda()
                labels = Variable(labels_cpu.transpose(0, 1).contiguous()).cuda()
            else:
                inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True) # 非阻塞，可与无依赖的操作并行执行。 只把数据放入gpu而不取出，访问时间会大大减少
        else: # 在cpu上训练
            labels = labels_cpu
        # .cuda的耗时并不多，并不算是耗时的主要原因，去掉也不会加速太多（来自一篇csdn博客）

        if trainer.dnn == 'lstman4':
            out, output_sizes = trainer.net(inputs, input_sizes) # 计算这一批minibatch的结果
            out = out.transpose(0, 1)  # TxNxH
            loss = trainer.criterion(out, labels_cpu, output_sizes, target_sizes) # 比较out和labels，计算loss 损失值
            torch.cuda.synchronize() # 等待当前设备上所有流中的所有内核完成。再进行下面的步骤
            loss = loss / inputs.size(0)  # average the loss by minibatch。这是怎么平均的？
        elif trainer.dnn == 'lstm' :
            hidden = trainer.net.init_hidden()
            hidden = lstmpy.repackage_hidden(hidden)
            outputs, hidden = trainer.net(inputs, hidden)
            tt = torch.squeeze(labels.view(-1, trainer.net.batch_size * trainer.net.num_steps))
            loss = trainer.criterion(outputs.view(-1, trainer.net.vocab_size), tt)
            torch.cuda.synchronize()
        else:
            # forward + backward + optimize
            #print(inputs.shape)
            if trainer.dataset == 'fashion-mnist': # 我加的
                inputs = inputs.float()
            outputs = trainer.net(inputs) # 前向传播
            if trainer.dataset == 'fashion-mnist':# 我加的
                labels = labels.long()
            loss = trainer.criterion(outputs, labels) # 计算损失值
            torch.cuda.synchronize() # 等待该设备上该流内核的完成

        if i >= warmup: # 预热完毕
            p.start() # 使参数is_profiling=true，否则hook没法用
        loss.backward() # 反向传播，计算当前（迭代次数的）梯度。前提：loss是tensor类型
        # pytorch每次调用.backward都会释放所有buffers。若模型中有多次backward，前一次backward存储在buffer中的梯度，会因为后一次调用backward而被释放
        # 使用 retain_graph=True，可以让前一次backward的梯度保存在buffer内，直到更新完成。
        if trainer.is_cuda:
            torch.cuda.synchronize() # 所有worker都结束BP
    end_time = time.time()
    tftb=(end_time - start_time) / (iteration + warmup)
    logger.info("tf+tb_comm=%f", tftb)
    layerwise_times, sum_total = p.get_layerwise_times() # 1行，1个平均值
    seq_keys = p.get_backward_seq_keys()
    p.stop() # 关闭hook
    return seq_keys[::-1], layerwise_times[::-1], p.get_backward_key_sizes()[::-1], tftb # -1表示将list倒序。都变成了从L到1层的顺序


class CommunicationProfiler(object):
    def __init__(self, comm_op, sync_op, sizes=None):
        self.comm_op = comm_op
        self.sync_op = sync_op
        self.sizes = sizes
        self._writer = SummaryWriter('./tb_log')

    def benchmark(self, num_iters=100):
        if self.sizes is None:
           # small_sizes = [8*1024*i for i in range(1, 64)] # 1K to 1M
            #small_sizes = [8*1000*i for i in range(1, 64)] # 1K to 1M
            small_sizes = [4*1024*i for i in range(1, 64)] # 1K to 1M
            large_sizes = []
            #large_sizes = [1024*1024*i for i in range(1, 8)] #[1024*1024*i for i in range(8)] # 1M to 512M
            # large_sizes = [1024*1024*i for i in range(150)] #[1024*1024*i for i in range(8)] # 1M to 150M
            # 10^6字节是5x10^5个浮点数
            sizes = small_sizes+large_sizes
            #logger.info("sizes= %f",list(sizes))
            print("communication figure")
        else:
            sizes = self.sizes
        warmup = 5
        size = 1024
        tensor = torch.rand(size).float().cuda() # tensor长度为size
        stime = time.time()
        for i in range(warmup):
            name = 'warmup-%d' % i
            h = self.comm_op(tensor, average=True, name=name)
            self.sync_op(h)
        etime = time.time()
        elapsed_times = []
        x = open("x.txt","w")
        y = open("y.txt","w")
        for s in sizes:
            tensor = torch.rand(s).float().cuda()
            torch.cuda.synchronize()
            stime = time.time()
            for i in range(num_iters):
                name = 'run-size%d-%d'% (s, i)
                h = self.comm_op(tensor, average=True, name=name)
                self.sync_op(h)
            etime = time.time()
            elapsed_times.append((etime-stime)/num_iters)
            #logger.info("here drawing")
            self._writer.add_scalar('communication', (etime-stime)/num_iters, s*4) # y-time, x-bytes
            x.write(f"{s*4}\n")
            y.write(f"{(etime-stime)/num_iters}\n")
            #print(s*4)
            if rank()==0:
                print((etime-stime)/num_iters)
        x.close()
        y.close()
        return sizes, elapsed_times