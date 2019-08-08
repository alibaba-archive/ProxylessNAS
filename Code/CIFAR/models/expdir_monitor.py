"""
Input: 'path' to the folder that contains configs and weights of a network
	Follow the configs and train the network
Output: 'Results' after training
"""
from __future__ import division
import json
import os

import torch
import numpy as np
import math


from models.networks.run_manager import RunConfig, RunManager
from models.networks import get_net_by_name

def generate_init_architecture_parameters():
    return np.ones([54,12,7])

def sample_candidate_index(architecture_parameters):
    softmax = torch.nn.Softmax()
    indexs = []
    for i in architecture_parameters:
        index_candidate_block = []
        for j in i:
            output = softmax(torch.Tensor(j))
            index_candidate_block.append(output.multinomial(1).numpy().tolist()[0])
        indexs.append(index_candidate_block)
    return indexs

def update_architecture_parameters(architecture_parameters, index_candidate_blocks, acc_results):
    M = len(acc_results)
    for block in range(len(architecture_parameters)):
        for layer in range(len(architecture_parameters[block])):
            sum_layer = sum([math.exp(i) for i in architecture_parameters[block][layer]])
            for op in range(len(architecture_parameters[block][layer])):
                grad = 0
                for m in range(M):
                    if op == index_candidate_blocks[m][block][layer]:
                        grad += 1 / M * acc_results[m] * (1 - math.exp(architecture_parameters[block][layer][op]) / sum_layer)
                    else:
                        grad += 1 / M * acc_results[m] * (-1 * math.exp(architecture_parameters[block][layer][op]) / sum_layer)
                # we want to maximize the object, so we use grad ascend
                architecture_parameters[block][layer][op] += grad
    return architecture_parameters

def set_init_candidate():
    index = np.random.randint(3,4, size=[54, 12])
    return index.tolist()



class ExpdirMonitor():
    def __init__(self, expdir):
        self.expdir = os.path.realpath(expdir)
        # os.makedirs(self.expdir, exist_ok=True)

    """ expdir paths """

    @property
    def logs_path(self):
        return '%s/logs' % self.expdir

    @property
    def save_path(self):
        return '%s/checkpoint' % self.expdir

    @property
    def output_path(self):
        return '%s/output' % self.expdir

    @property
    def run_config_path(self):
        return '%s/run.config' % self.expdir

    @property
    def net_config_path(self):
        return '%s/net.config' % self.expdir

    @property
    def init_path(self):
        return '%s/init' % self.expdir

    """ methods for loading """

    def load_run_config(self, print_info=False, dataset='cifar10'):
        if os.path.isfile(self.run_config_path):
            run_config = json.load(open(self.run_config_path, 'r'))
        else:
            run_config = RunConfig.get_default_run_config(dataset)
        run_config = RunConfig(**run_config)
        if print_info:
            print('Run config:')
            for k, v in run_config.config.items():
                print('\t%s: %s' % (k, v))
        return run_config

    def load_net(self, print_info=False):
        assert os.path.isfile(self.net_config_path), 'No net configs found in <%s>' % self.expdir
        net_config_json = json.load(open(self.net_config_path, 'r'))
        if print_info:
            print('Net config:')
            for k, v in net_config_json.items():
                if k != 'blocks':
                    print('\t%s: %s' % (k, v))
        ### NOTE
        net = get_net_by_name(net_config_json['name']).build_from_config(net_config_json)
        return net

    def get_op_index_from_txt(self):
        with open('./op_list.txt', "r") as f:
            data = f.readlines()
        candidates = ['dilated_conv', 'Identity', '3x3_conv', '5x5_conv', '7x7_conv', '3x3_avg_pool', '3x3_max_pool']
        indexs = []
        for i in range(len(data)):
            index = []
            for j in data[i][:-1].split(' '):
                index.append(candidates.index(j))
            indexs.append(index)
        return indexs

    def load_samplenet(self, indexs):
        assert os.path.isfile(self.net_config_path), 'No net configs found in <%s>' % self.expdir
        net_config_json = json.load(open(self.net_config_path, 'r'))
        net = get_net_by_name(net_config_json['name']).build_from_config(net_config_json, True, indexs)
        return net

    def load_init(self):
        if os.path.isfile(self.init_path):
            if torch.cuda.is_available():
                checkpoint = torch.load(self.init_path)
            else:
                checkpoint = torch.load(self.init_path, map_location='cpu')
            return checkpoint
        else:
            return None

    def train_samplent(self):
        indexs = self.get_op_index_from_txt()
        dataset = 'cifar10'
        run_config = self.load_run_config(print_info=True, dataset=dataset)
        net = self.load_samplenet(indexs)
        run_manager = RunManager(self.expdir, net, run_config, out_log=True)
        run_manager.train_samplenet()

    def train_supernet(self):
        dataset = 'cifar10'
        run_config = self.load_run_config(print_info=True, dataset=dataset)
        net = self.load_net(print_info=True)
        run_manager = RunManager(self.expdir, net, run_config, out_log=True)
        run_manager.train_supernet()
