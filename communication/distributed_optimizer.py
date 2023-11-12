# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allgather_async
from horovod.torch.mpi_ops import broadcast_async_
from horovod.torch.mpi_ops import synchronize
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import init, broadcast

import time
import torch
import numpy as np
import utils

import collections
import settings
from settings import logger, ADAPTIVE_MERGE, DEBUG

from profiling import CommunicationProfiler
from sklearn.linear_model import LinearRegression


class _DistributedOptimizer(torch.optim.Optimizer): # 继承了torch.optim.Optimizer类（基类）
    def __init__(self, params, named_parameters, compression, seq_layernames=None, layerwise_times=None, norm_clip=None, threshold=0, tb_writer=None, gradient_path=None):
        super(self.__class__, self).__init__(params)
        self._compression = compression
        self._density = 1
        self._profiling = False
        self._seq_layernames = seq_layernames
        self._layerwise_times = layerwise_times
        self._original_layerwise_times_kv = None
        self._norm_clip = norm_clip
        self._threshold = threshold
        #self._writer = writer
        self._writer = tb_writer
        self._gradient_path = gradient_path
        self.alpha = None
        self.beta = None
        self.gamma = None

        if self._layerwise_times is not None and self._seq_layernames is not None:
            self._original_layerwise_times_kv = dict(zip(self._seq_layernames, self._layerwise_times))
        self._compression_timers = {} # compression
        self._allreduce_timers = {} # allreduce times
        self._update_times = {} # allreduce times
        self.train_epoch = 0
        self.train_iter = 0
        self._dynamic_densities = None
        self._layerwise_compressors= None

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        self._named_parameters = {k: v for k, v
                                in named_parameters}
        if self._seq_layernames is not None:
            self._sequential_keys = self._seq_layernames
        else:
            self._sequential_keys = [k for k, v in named_parameters]

        self.size_commtime_dict = None

        self._debug_seq_keys = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        if len(named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                     in sorted(named_parameters)}
        else:
            self._parameter_names = {v: 'allreduce.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}

        self.p = []
        self._generate_merged_parameters() # 初始化时 产生合并方案
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set() # 无序不重复元素集
        self.local = False
        self._hook_checked_idx = 0

        if size() > 1:
            self._register_hooks() # 注册hook，当调用make_hook时，hook发生作用



    def _benchmark_communication(self):
        logger.info('Benchmarking communication performance...')
        comm_profiler = CommunicationProfiler(allreduce_async_, synchronize)
        sizes, times = comm_profiler.benchmark(num_iters=10)
        def _fit_linear_function(x, y):
            X = np.array(x).reshape((-1, 1)) * 4
            Y = np.array(y)
            model = LinearRegression()
            model.fit(X, Y)
            alpha = model.intercept_
            beta = model.coef_[0]
            return alpha, beta
        alpha, beta = _fit_linear_function(sizes, times)
        logger.info("alpha, beta= %f, %f", alpha, beta)
        #self._writer.add_scalar('communication', times, sizes)
        #self._writer.add_scalar()
        self.alpha = alpha
        self.beta = beta
        alpha_tensor = torch.ones(1) * alpha
        beta_tensor = torch.ones(1) * beta
        alpha_tensor = broadcast(alpha_tensor, root_rank=0)
        beta_tensor = broadcast(beta_tensor, root_rank=0)
        if rank() != 0:
            self.alpha = float(alpha_tensor[0])
            self.beta = float(beta_tensor[0])
        logger.info('[rank:{}] Communication performance fitted with f(p)=a+b*p, where a={} and b={}'.format(rank(), self.alpha, self.beta))

    def _register_hooks(self): # 在_init_时就被调用了，初始化hook
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad: # 查看可训练参数 （如果没有被冻结
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p)) # 关键语句
                    self._grad_accs.append(grad_acc)

    def _generate_groups_with_threshold(self, threshold):
        sizes = [self._named_parameters[k].data.numel() for k in self._sequential_keys][::-1] # reverse order
        self._sizes = sizes
        sub_size = 0
        groups = []
        group = []
        key_groupidx_maps = {}
        idx = 0
        for k in self._sequential_keys[::-1]:
            numel = self._named_parameters[k].data.numel()
            sub_size += numel
            key_groupidx_maps[k] = idx
            if sub_size < threshold:
                group.append(k)
            else:
                idx += 1
                group.append(k)
                groups.append(group)
                group = []
                sub_size = 0
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps

    def _generate_groups_stwfbp(self):
        #num_of_workers = size()
        num_of_workers = 2
        """
        p_alpha_beta_56Gbps = { # latency_transmission time
                16: (0.00023583677659915685, 4.0594787739537565e-10),
                8: (9.75367204301171e-05, 3.0568230536676206e-10),
                4: (4.204298980348825e-05, 2.0589360830118177e-10),
                2: (2.554691138304671e-06, 9.837548167872609e-11)
                }
        """
        p_alpha_beta_gama_10Gbps = {  # p是机器数
            2: (2.10295057210933e-4, 9.987246076e-10, 5.9392e-5)# 参考ASC 8-node
            #2:（, , 0）# PS下的alpha, beta
        }
        p_mg_wfbp = {
            2: (2.554691138304671e-06, 9.837548167872609e-11 , 0)
        }

        if self.alpha is not None:
            alpha, beta = self.alpha, self.beta
        else:
            if settings.CONNECTION == '10GbE':
                alpha, beta, gamma = p_alpha_beta_gama_10Gbps[num_of_workers]
                #alpha, beta, gamma = p_mg_wfbp[num_of_workers]
            else:
                alpha, beta, gamma = 1,1,1
                #alpha, beta = p_alpha_beta_56Gbps[num_of_workers]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        nbytes = 2 if settings.FP16 else 4 # 精度

        def __calculate_comm_start(tc, tb, taob, L):  # 已修改
            taoc = [0] * L
            taoc[L - 1] = taob[L - 1] + tb[L - 1]
            for l in range(L-1)[::-1]:
                taoc[l] = max(taoc[l+1] + tc[l+1], taob[l] + tb[l])
            return taoc

        def __maximum_L_trans(cb, taoc, l, first): # 计算切分后两部分的大小，个数 不是字节数
            if first: # 小组内第一个
                trans_bytes = (cb - taoc[l] - self.alpha) / self.beta
            else: # 小组内中间层
                trans_bytes = (tb[l-1] - self.alpha) / self.beta
            n_M = trans_bytes / nbytes # nbytes是否确实是2呢？float
            return n_M

        def __merge(taob, tc, p, l): # 第L层的梯度通信要与第L-1层合并，那梯度开始通信时间，要等第L-1层的梯度计算完
            tc[l] = 0
            p[l-1] = p[l-1]+p[l] # P是第L层的参数数量。若L层是合并层，把L层的梯度与L-1层的梯度合并通信。
            p[l] = 0
            if self.size_commtime_dict is not None:
                tc[l-1] = self.size_commtime_dict[l-1]
            else: # None
                tc[l-1] = utils.predict_allreduce_time_with_size(alpha, beta, p[l-1]*nbytes, num_of_workers) # allreduce 这么多bytes的时间

        sizes = [self._named_parameters[k].data.numel() for k in self._seq_layernames] # 每层参数个数
        seq_layernames = self._seq_layernames
        if not utils.check_unique(seq_layernames):
            raise ValueError
        self._sizes = sizes
        p = sizes[:] # sizes[]  存储了model.每层参数的个数
        ori_p = sizes[:]
        L = len(sizes) # 总层数

        ori_tc = [utils.predict_allreduce_time_with_size(alpha, beta, s*nbytes, num_of_workers) for s in sizes] # 对于每一层，计算各层的tc（l）
        tc = [utils.predict_allreduce_time_with_size(alpha, beta, s*nbytes, num_of_workers) for s in sizes] # 对于每一层，计算各层的tc（l）
        tb = list(self._layerwise_times) # tb=每次迭代中反向传播的时间。这个layerwise time 好像是在benchmark中计算出来的各层BP时间
        #print(tc)
        #print(p)
        taob = [0]*L # 开始计算梯度的时间戳
        for l in range(0,L-1)[::-1]: # 从L-2到0
            taob[l] = taob[l+1] + tb[l+1] # 论文公式6

        ori_taoc = __calculate_comm_start(tc, tb, taob, L) # 第L层开始通信的时间戳
        taoc = __calculate_comm_start(tc, tb, taob, L)
        if rank() == 0:
            logger.info('tc sum: %f', np.sum(tc))
        groups = [] # [[group],[],[]] 统计各层情况，在一个小[]中的是相合并的层
        group = [] # 相合并的层
        idx = 0
        key_groupidx_maps = {}
        l = L-1
        key = seq_layernames[l]
        key_groupidx_maps[key] = idx
        #pre_merged = False
        for l in range(1, L)[::-1]:  # 从第L-1层到第1层
            key = seq_layernames[l]
            group.append(key)
            key_groupidx_maps[key] = idx
            current_taob = taob[l - 1] + tb[l - 1]  # BP中，第L-2层开始计算的时间
            merged = False
            if current_taob < taoc[l] + tc[l]:  # 第L层的通信时间没有被完全覆盖
                if taoc[l] > current_taob:  # 第L层开始通信的时刻，大于 第L-1层梯度算完的时刻
                    __merge(taob, tc, p, l)
                    taoc = __calculate_comm_start(tc, tb, taob, L)  # tc已被更新 tc[L-1]。再经过max()比较，更新taoc列表。这可以算缺点不，每次有更新都要循环+判断一下
                    merged = True
                else:
                    t_wait = current_taob - taoc[l]  # 不梯度合并，而多出的时间
                    t_saved = alpha  # startup time，梯度合并可以节省的启动时间 （ 但节省的时间也不都来自于startup time吧？）
                    if t_wait < t_saved:  # 值得合并.[2019INFOCOM/TPDS]
                        __merge(taob, tc, p, l)
                        taoc = __calculate_comm_start(tc, tb, taob, L)
                        merged = True
            if not merged:  # if False
                idx += 1
                groups.append(group)
                group = []
        l = 0
        key = seq_layernames[l]
        #self.mp_curLname_transNumel[key] = p[0]
        key_groupidx_maps[key] = idx # name与index的映射，同一小组内的idx相同
        group.append(key)
        if len(group) > 0:
            groups.append(group)

        wait_time = []
        for g in groups:
            first_key = g[0]
            left_index = key_groupidx_maps[first_key]
            logger.info("fist_index=%s", left_index)
            length = len(g)
            last_key = g[length - 1]
            right_index = key_groupidx_maps[last_key]
            logger.info("fist_index=%s", right_index)
            wait_time.append(taoc[right_index] - taoc[left_index])  # 没写错，先算出来的taoc是不变的
        if len(groups[0]) == L:
            wait_time.append(taoc[0] - taob[L - 1] - tb[l - 1])

        if rank() == 0:
            logger.info('Predicted non-overlapped time: %f', taoc[0]+tc[0]-(taob[0]+tb[0]))
            logger.info('Predicted tb+tc= %f', taoc[0]+tc[0])
            logger.info('MG tc sum: %f', np.sum(tc)) # tc是直接被算出来的，根据α β γ
            logger.info('before optimizing, wait time: %f', np.sum(wait_time))

        logger.info("before optimizing, non-overlapped time= %f", taoc[0] + tc[0] - (taob[0] + tb[0]))
        # t: 每个小组浪费的时间
        t_wasted = 0
        optimizer_layer_transNum = [0] * L # 若为0 则按小组发送，若有值n，则发送n个参数
        #self.optimized_layer = []
        def optimize(t, tc, p, l, first_l):
            optimizer_layer_transNum[l] = int((t - alpha) / (beta * nbytes))  # p[l]
            p[first_l] = p[first_l] - optimizer_layer_transNum[l]
            tc[l] = t
            tc[first_l] = utils.predict_allreduce_time_with_size(alpha, beta, p[first_l]*nbytes, num_of_workers)

        for g in groups: # 每个合并组
            if len(g) > 1:
                last_l = self._sequential_keys.index(g[0])
                first_l = self._sequential_keys.index(g[len(g) - 1]) # [first_l,last_l]
                if (last_l == L - 1 and first_l==0) or last_l != L-1: # 所有层可以都被合并到一层去
                    logger.info("optimizing...")
                    # 跳过最顶层小组的正常发送
                    t = taoc[first_l] - taoc[last_l]  # 论文公式，浪费的空闲时间
                    #logger.info("t= %f",t)
                    #t_wasted += t
                    #offset = 0
                    #k = g[0]

                    if t > alpha: # 这层一定是传不完的
                        optimize(t, tc, p, last_l, first_l)
                        #taoc = __calculate_comm_start(tc, tb, taob, L)
                        #self.optimized_layer_offsets[k] = offset

            self.optimizer_layer_transNum = optimizer_layer_transNum
        self.p = p
        logger.info('MAX tc sum: %f', np.sum(tc))  # tc是直接被算出来的，根据α β γ
        logger.info("after optimizing, non-overlapped time= %f",taoc[0]+tc[0]-(taob[0]+tb[0]))
        #logger.info("after optimizing, wait time=%f",)
        return groups, key_groupidx_maps

    def _generate_merged_parameters(self): # init()中被调用
        self._store_grouped_parameters = {}
        self._rev_grouped_parameters = {}
        #self.optimized_layer_offsets = {}
        #sizes = [self._named_parameters[k].data.numel() for k in self._seq_layernames]  # 每层参数个数
        self.optimizer_layer_transNum = []
       # self.optimized_layer = []
        self.idle_group_time = {}

        if ADAPTIVE_MERGE and self._layerwise_times is not None:
            #ori_taoc, taoc, groups, key_groupidx_maps, p = self._generate_groups_stwfbp() # 获得合并分组[[],[],[]]
            groups, key_groupidx_maps = self._generate_groups_stwfbp() # 获得合并分组[[],[],[]]
            logger.info("mg-wfbp groups= %d",len(groups))
            logger.info("have generated merged group")
        else:
            groups, key_groupidx_maps = self._generate_groups_with_threshold(self._threshold)
            logger.info("not generate merged group")
        logger.info('# of parameters: %d', np.sum(self._sizes))
        #logger.info('# of parameters: %d', np.sum(self._sizes))
        logger.info('Total number of tensors: %s', len(self._sizes))
        logger.info('Merged Number of groups: %s', len(groups))
        logger.info("layers num = %d",len(self._sequential_keys))
        new_keys = []
        self._merged_parameter_offsets = {}
        self._layerwise_compressors = None
        self._layerwise_compressors = {}
        self.groupidx_newkey = {}
        self._key_groupidx_maps = key_groupidx_maps


        for g in groups: # 可能 每层一组
            sub_size_ori = 0  # 每层原始参数-size
            offsets_ori = [] # p(l)的offset
            device = self._named_parameters[g[0]].device
            dtype = self._named_parameters[g[0]].dtype
            for k in g: # 统计这一合并小组，需要多少空间
                numel = self._named_parameters[k].data.numel()
                offsets_ori.append(sub_size_ori)
                sub_size_ori += self._named_parameters[k].numel()
            new_key = ':'.join(g)
            new_keys.append(new_key)
            # requires_grad=False 表示不会再对此值求导
            # t = [1,sub_size]

            t = torch.zeros(sub_size_ori, device=device, dtype=dtype, requires_grad=False) # 返回一个shape为sub_size,数值类型为dtype，值都是0的tensor
            self._store_grouped_parameters[new_key] = t # new_key 是合并后一个合并组的名字。虽然t的值为0，但不同t的大小不一样
            self._merged_parameter_offsets[new_key] = offsets_ori
            self.groupidx_newkey[self._key_groupidx_maps[g[0]]] = new_key
            self._rev_grouped_parameters[new_key] = t # 给synchronize使用分配了位置,mp(l)

        self._groups = groups
        self._groups_flags = []
        for g in self._groups:
            flags = []
            for k in g:
                flags.append(0)
            self._groups_flags.append(flags) # 每一层都有一个flag标记位

    def _push_to_buffer(self, name, tensor): # 在make_hook(自动调用）中被调用,当要梯度聚合时被调用
        # 把获取到的该层梯度，放进待传输缓冲区中，并返回当前应发送的张量 或 none
        with torch.no_grad(): # 只有handle拿到值了，才能填充
            if len(self._groups) == len(self._sequential_keys):
                new_tensor = tensor.data.view(-1) # 张量变成一维
                return name, new_tensor # 若分组都是单层
            group_idx = self._key_groupidx_maps[name] # 获取该层所在的分组
            g = self._groups[group_idx] # 所在分组[]
            new_key = ':'.join(g)
            layer_idx = g.index(name) # 是该小组的第几个

            # 把该层全部梯度放进来
            numel = tensor.data.numel()  # ori_p[l]
            offset = self._merged_parameter_offsets[new_key][layer_idx] # p(l) offset
            self._store_grouped_parameters[new_key].data[offset:offset+numel].copy_(tensor.view(-1))
            self._groups_flags[group_idx][layer_idx] = 1

            if len(g) == 1: #如果是单层，直接发送
                return name, self._store_grouped_parameters[new_key]

            cur_l = self._sequential_keys.index(name)

            if self.optimizer_layer_transNum[cur_l] > 0: # 这是优化层，直接发送优化部分transNum
            #if name in self.optimized_layer:
                #offset = self.optimized_layer_offsets[name]
                offset = 0
                numel = self.optimizer_layer_transNum[cur_l]
                new_tensor = torch.zeros(numel, device=self._named_parameters[g[0]].device,
                                dtype=self._named_parameters[g[0]].dtype,
                                requires_grad=False)
                new_tensor.copy_(self._store_grouped_parameters[new_key].data[offset:offset + numel].view(-1))

                return name, new_tensor

            for idx in self._groups_flags[group_idx]:
                if idx == 0: # 等小组存满
                    return name, None

            # 合并发送小组梯度
            first_l = self._sequential_keys.index(name)
            #numel = self._store_grouped_parameters[new_key].data.numel() - offset
            numel = self.p[first_l]
            new_tensor = torch.zeros(numel, device=self._named_parameters[g[0]].device,
                                     dtype=self._named_parameters[g[0]].dtype,
                                     requires_grad=False)
            n = self._store_grouped_parameters[new_key].data.numel()
            new_tensor.copy_(self._store_grouped_parameters[new_key].data[n-numel:].view(-1))
            return new_key, new_tensor


    def _pull_from_buffer(self, name, merged_tensor): # <group_name, group_merged_p>
        if len(self._groups) == len(self._sequential_keys): # 每组一层
            shape = self._named_parameters[name].data.shape
            return {name: merged_tensor.view(shape)}
        offsets = self._merged_parameter_offsets[name] # 原始pl
        g = name.split(':') # 从合并小组中，把每层分离出来
        group_idx = self._key_groupidx_maps[g[0]]
        #self._groups_flags[group_idx] = [0]*len(self._groups_flags[group_idx]) # flag 清空
        tensors = {}
        for i, k in enumerate(g): # g的第i个，是k, k=l_name
            offset = offsets[i] # 原始pl的偏置量
            original_tensor = self._named_parameters[k]
            numel = original_tensor.numel() # 复原原始参数的位置
            tensors[k] = merged_tensor.data[offset:offset+numel].view(original_tensor.shape)
        return tensors

    def _allreduce_grad_async(self, p, name): # mp(l),name(l)
        tensor = p.data.view(-1)
        allreduce_name = name
        if len(name) > 100:
            allreduce_name = name[0:50]+'...'+name[50:100] # 缩写
        handle = allreduce_async_(tensor, average=True, name=allreduce_name) # A name of the reduction operation.
        return handle, None


    def check_hooked_tensor_sequence(self, name):
        if self._seq_layernames is None:
            return
        ntensors = len(self._seq_layernames) # 层数
        idx = self._seq_layernames.index(name)
        if idx == ntensors-self._hook_checked_idx-1:
            self._hook_checked_idx += 1
            if idx == 0:
                self._hook_checked_idx = 0
        else:
            logger.info('Hook checked error, name: %s should be in the index of %d, which it runs at %d',
                    name, self._hook_checked_idx, idx)
            raise

    def _make_hook(self, p): # p是这一层的参数
        def hook(*ignore):
            assert p not in self._handles
            assert not p.grad.requires_grad
            if not self.local: # if False，要梯度聚合了
                name = self._parameter_names.get(p)
                self.check_hooked_tensor_sequence(name)
                # push_to_buffer: 将梯度放到相应的位置上去，并将flag标志为1，当该合并小组内所有层的梯度都放进来了，发送该小组的梯度
                new_name, new_tensor = self._push_to_buffer(name, p.grad.data) # p.grad.data是一维的.层名字，本层发送内容
                # 对送上去的mp(l)梯度，allreduce求平均
                if new_tensor is not None:
                    handle, ctx = self._allreduce_grad_async(new_tensor, new_name) # 这层l算完了，就可以发送mp(l)
                    self._handles[new_tensor] = (new_name, handle, ctx, 1) # self._handles[mp(l)] = (handle,None,1)
        return hook

    def _check_group(self,groupidx):
        for idx in self._groups_flags[groupidx]:
            if idx == 0:
                return False
        return True

    def _get_mp_info(self,l_name):

        group_idx = self._key_groupidx_maps[l_name]  # 该层所在小组序号  l_name=None
        group_name = self.groupidx_newkey[group_idx]
        g = self._groups[group_idx]  # 该层所在小组list
        self._groups_flags[group_idx][g.index(l_name)] = 1
        #numel = p.data.numel()
        offset_list = self._merged_parameter_offsets[group_name]  # list
        layer_idx = g.index(l_name)
        offset = offset_list[layer_idx]
        return group_idx, group_name

    def synchronize(self):
        for p, value in self._handles.items():  # p=new_tensor=mp, value=（handle,ctx,1） 对每个mp
           # 发送小组内的梯度
            new_name, handle, ctx, density = value # l_name
            output = synchronize(handle)  # 等所有process都返回了handle
            p.set_(output) # 同步、更新的梯度

        # 等小组都放进来了再划分
        if len(self._groups) != len(self._sequential_keys):
            for merged_p, value in self._handles.items(): # 送上去什么形状，送下来还是什么形状，所以要再根据offset划分开
                new_name, handle, ctx, density = value # 发送上去的name
                g = new_name.split(':') # 可能单层，可能小组
                group_idx, group_name = self._get_mp_info(g[0])
                numel = merged_p.data.numel()
                if len(g) == 1:
                    layer_idx = self._groups[group_idx].index(g[0])
                    if numel == self._named_parameters.get(g[0]).size: # 单层（可能是优化层）是完整的一层
                        offset = self._merged_parameter_offsets[group_name][layer_idx]
                        self._rev_grouped_parameters[group_name].data[offset:offset + numel].copy_(merged_p)
                        self._groups_flags[group_idx][layer_idx] = 0
                    else: # 这层没发完
                        #cur_l = self._sequential_keys.index(g[0])
                        last_l = self._sequential_keys.index(self._groups[group_idx][0]) # 118
                        offset = 0
                        for l in range(last_l-layer_idx+1, last_l+1)[::-1]: #
                            if self.optimizer_layer_transNum[l] > 0:
                                offset += self.optimizer_layer_transNum[l]
                            else:
                                break
                        self._rev_grouped_parameters[group_name].data[offset:offset+numel].copy_(merged_p)
                        self._groups_flags[group_idx][layer_idx] = 0
                else: # 小组
                    offset = self._rev_grouped_parameters[group_name].data.numel() - merged_p.data.numel()
                    self._rev_grouped_parameters[group_name].data[offset:].copy_(merged_p)
                    self._groups_flags[group_idx] = [0] * len(self._groups_flags[group_idx])

                if np.sum(self._groups_flags[group_idx]) == 0 : # 如果小组齐了
                    tensors = self._pull_from_buffer(group_name, self._rev_grouped_parameters[group_name])  # 将合并形状 拆分成 原本的形状
                    for n in tensors:
                        # 可根据n，知道它的new_key，还需要把它拼接完整，再赋值p.grad
                        p = self._named_parameters.get(n)  # 获取这个梯度/张量中的参数，设置
                        if settings.FP16:  # 精度
                            p.grad.set_(tensors[n].data.type(p.grad.type()))  # 将平均后的梯度，暂存
                        else:
                            p.grad.set_(tensors[n].data)

        self.train_iter += 1
        self._handles.clear()
        self._print_profiling() # 打印平均值


    def _print_profiling(self):
        if self._profiling and rank() == 0 and len(self._allreduce_timers.keys()) > 0 and len(self._allreduce_timers.get(self._allreduce_timers.keys()[0], [])) ==  40:
            cps = self._compression_timers # compression
            ars = self._allreduce_timers # allreduce times
            ups = self._update_times # update times
            r = rank()
            tcp = 0.0; tar = 0.0; tup = 0.0; total=0.0
            for k in cps:
                acp = np.mean(cps[k])
                tcp += acp
                aar = np.mean(ars[k])
                tar += aar
                aup = np.mean(ups[k])
                tup += aup
            total = tcp+tar+tup
            logger.info('[%d]: Total compress: %f, allreduce: %f, update: %f, total: %f', r, tcp, tar, tup, total)
            cps.clear()
            ars.clear()
            ups.clear()


    def step(self, closure=None): # 确保在update之前，allreduce操作已完成
        if not self.local:
            self.synchronize() # 当要聚合时，开始同步
        return super(self.__class__, self).step(closure) # 用的是本类中的方法
               # 解决多重继承时父类的查找问题 （明确指定了调用的方法是哪个类中的方法）



def DistributedOptimizer(optimizer, named_parameters=None, compression=None, density=0.001, seq_layernames=None, layerwise_times=None, norm_clip=None, threshold=0, writer=None, gradient_path=None):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to （在将梯度应用到模型权重之前，使用allreduce求平均梯度值）
    average gradient values before applying gradients to model weights.

    Allreduce operations are executed after each gradient is computed by `loss.backward()` （在loss.backward()并行计算完梯度后，执行allreduce操作）
    in parallel with each other. The `step()` method ensures that all allreduce operations are （step()确保了在梯度应用到模型权重之前，所有allreduce操作已完成）
    finished before applying gradients to the model.

    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.

    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    def __init__(self):
        self.trans_params_under_prevL = {}
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    return cls(optimizer.param_groups, named_parameters, compression, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=None, threshold=threshold, tb_writer=writer, gradient_path=gradient_path)


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        handle = broadcast_async_(p, root_rank, name)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()
