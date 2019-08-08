from __future__ import division
import random

from models.utils import *
from models.modules.layers import set_layer_from_config, set_candidate_for_cell

### NOTE
def set_tree_node_from_config(tree_node_config, count, isSamplenet=False, index=None):
    # this is a function to get cell from config
    name2tree_node = {
        NormalTreeNode.__name__: NormalTreeNode,
    }

    tree_node_name = tree_node_config.pop('name')
    tree_node = name2tree_node[tree_node_name]
    return tree_node.build_from_config(tree_node_config, count, isSamplenet, index)

class Count(object):
    def __init__(self):
        self.count = 0


class NormalTreeNode(BasicUnit):

    def __init__(self, edges, child_nodes, in_channels, out_channels,
                 split_type='copy', merge_type='add', use_avg=True, skip_node_bn=False, path_drop_rate=0):
        super(NormalTreeNode, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_type = split_type
        self.merge_type = merge_type

        self.use_avg = use_avg
        self.skip_node_bn = skip_node_bn

        self.path_drop_rate = path_drop_rate

        assert len(edges) == len(child_nodes)

        """ add modules """
        self.edges = nn.ModuleList(edges)
        self.child_nodes = nn.ModuleList(child_nodes)
        self.edges_index = [0, 0]
        self.infe_count = 0

        # branch batch norm (skip node bn)
        if self.skip_node_bn:
            branch_bns = []
            for out_dim in self.out_dim_list:
                branch_bns.append(nn.BatchNorm2d(out_dim))
        else:
            branch_bns = [None] * self.child_num
        self.branch_bns = nn.ModuleList(branch_bns)

    def set_candidate_for_nodes(self, candidate_index, flag=0):
        if flag != 0:
            self.edges_index = [candidate_index.pop(0), candidate_index.pop(0)]
        for node in self.child_nodes:
            if node:
                if len(self.child_nodes) == 1:
                    flag = 0
                else:
                    flag = 1
                node.set_candidate_for_nodes(candidate_index, flag)
    """ tree-related properties """

    @property
    def child_num(self):
        return len(self.edges)

    @property
    def in_dim_list(self):
        if self.split_type == 'copy':
            in_dim_list = [self.in_channels] * self.child_num
        elif self.split_type == 'split':
            in_dim_list = get_split_list(self.in_channels, self.child_num)
        else:
            assert self.child_num == 1
            in_dim_list = [self.in_channels]
        return in_dim_list

    @property
    def out_dim_list(self):
        if self.merge_type == 'add':
            out_dim_list = [self.out_channels] * self.child_num
        elif self.merge_type == 'concat':
            out_dim_list = get_split_list(self.out_channels, self.child_num)
        else:
            assert self.child_num == 1
            out_dim_list = [self.out_channels]
        return out_dim_list

    """ tree-related calculation methods """

    def get_node(self, path2node):
        node = self
        for step in path2node:
            node = node.child_nodes[step]
        return node

    def allocation_scheme(self, x):
        if self.split_type == 'copy':
            child_inputs = [x] * self.child_num
        elif self.split_type == 'split':
            child_inputs, _pt = [], 0
            for seg_size in self.in_dim_list:
                seg_x = x[:, _pt:_pt + seg_size, :, :].contiguous()  # split in the channel dimension
                child_inputs.append(seg_x)
                _pt += seg_size
        else:
            child_inputs = [x]
        return child_inputs

    def merge_scheme(self, child_outputs):
        if self.merge_type == 'concat':
            output = torch.cat(child_outputs, dim=1)
        elif self.merge_type == 'add':
            output = list_sum(child_outputs)
            if self.use_avg:
                output = output / self.child_num
        else:
            assert len(child_outputs) == 1
            output = child_outputs[0]
        return output

    @staticmethod
    def path_normal_forward(x, edge=None, child=None, branch_bn=None, use_avg=False):
        if edge is not None:
            x = edge(x)
        edge_x = x
        if child is not None:
            x = child(x)
        if branch_bn is not None:
            x = branch_bn(x)
            x = x + edge_x
            if use_avg:
                x = x / 2
        return x

    def path_drop_forward(self, x, branch_idx):
        edge_name = ['ConvLayer', 'IdentityLayer', 'DepthConvLayer', 'PoolingLayer']

        if self.edges[branch_idx].__class__.__name__ in edge_name:
            edge, child, branch_bn = self.edges[branch_idx], self.child_nodes[branch_idx], self.branch_bns[branch_idx]
        else:
            edge, child, branch_bn = self.edges[branch_idx][self.edges_index[branch_idx]], self.child_nodes[branch_idx], self.branch_bns[branch_idx]
        if self.path_drop_rate > 0:
            if self.training:
                # train
                p = random.uniform(0, 1)
                drop_flag = p < self.path_drop_rate
                if drop_flag:
                    batch_size = x.size()[0]
                    feature_map_size = x.size()[2:4]
                    stride = edge.__dict__.get('stride', 1)
                    out_channels = self.out_dim_list[branch_idx]
                    padding = torch.zeros(batch_size, out_channels,
                                          feature_map_size[0] // stride, feature_map_size[1] // stride)
                    if x.is_cuda:
                        padding = padding.cuda()
                    path_out = torch.autograd.Variable(padding, requires_grad=False)
                else:
                    path_out = self.path_normal_forward(x, edge, child, branch_bn, use_avg=self.use_avg)
                    path_out = path_out / (1 - self.path_drop_rate)
            else:
                # test
                path_out = self.path_normal_forward(x, edge, child, branch_bn, use_avg=self.use_avg)
        else:
            path_out = self.path_normal_forward(x, edge, child, branch_bn, use_avg=self.use_avg)
        return path_out

    """ abstract methods defined in BasicUnit """

    def forward(self, x):
        child_inputs = self.allocation_scheme(x)

        child_outputs = []
        for branch_idx in range(self.child_num):
            path_out = self.path_drop_forward(child_inputs[branch_idx], branch_idx)
            child_outputs.append(path_out)

        output = self.merge_scheme(child_outputs)
        return output

    @property
    def unit_str(self):
        if self.child_num > 0:
            children_str = []
            for _i, child in enumerate(self.child_nodes):
                child_str = None if child is None else child.unit_str
                children_str.append('%s=>%s' % (self.edges[_i].unit_str, child_str))
            children_str = '[%s]' % ', '.join(children_str)
        else:
            children_str = None
        return 'T(%s-%s, %s)' % (self.merge_type, self.split_type, children_str)

    @property
    def config(self):
        child_configs = []
        for child in self.child_nodes:
            if child is None:
                child_configs.append(None)
            else:
                child_configs.append(child.config)
        edge_configs = []
        for edge in self.edges:
            if edge is None:
                edge_configs.append(None)
            else:
                edge_configs.append(edge.config)
        return {
            'name': NormalTreeNode.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'split_type': self.split_type,
            'merge_type': self.merge_type,
            'use_avg': self.use_avg,
            'skip_node_bn': self.skip_node_bn,
            'path_drop_rate': self.path_drop_rate,
            'edges': edge_configs,
            'child_nodes': child_configs,
        }

    @staticmethod
    def build_from_config(config, count, isSamplenet=False, index=None):
        # print(index)
        edges = []
        ### NOTE
        edge_configs = config.pop('edges')
        for edge_config in edge_configs:
        # for edge_config in config.pop('edges'):
            if edge_config is None:
                edges.append(None)
            else:
                if len(edge_configs) == 2:
                    count.count += 1
                if count.count <= 2:
                    edges.append(set_layer_from_config(edge_config))
                else:
                    if isSamplenet:
                        edges.append(set_candidate_for_cell(edge_config)[index.pop(0)])
                    else:
                        edges.append(set_candidate_for_cell(edge_config))

        child_nodes = []
        for child_config in config.pop('child_nodes'):
            if child_config is None:
                child_nodes.append(None)
            else:
                child_nodes.append(set_tree_node_from_config(child_config, count, isSamplenet, index))
        return NormalTreeNode(edges=edges, child_nodes=child_nodes, **config)

    def get_flops(self, x):
        child_inputs = self.allocation_scheme(x)

        child_outputs = []
        flops = 0
        for branch_idx in range(self.child_num):
            edge, child, branch_x = self.edges[branch_idx], self.child_nodes[branch_idx], child_inputs[branch_idx]
            if edge is not None:
                edge_flop, branch_x = edge.get_flops(branch_x)
                flops += edge_flop
            if child is not None:
                child_flop, branch_x = child.get_flops(branch_x)
                flops += child_flop
            child_outputs.append(branch_x)
        output = self.merge_scheme(child_outputs)
        return flops, output
