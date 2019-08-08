from copy import deepcopy

import torch.nn as nn
from models.utils import *


def set_layer_from_config(layer_config):
    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
    }
    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)

def set_candidate_for_cell(layer_config):
    # order: dilation Identity [depth_conv 3 5 7] avg_pool max_pool
    layer_name = layer_config.pop('name')
    candidates = nn.ModuleList()
    if layer_name == 'DepthConvLayer':
        # add a dilation
        config_copy = deepcopy(layer_config)
        config_copy['kernel_size'] = 3
        config_copy['dilation'] = 2
        candidates.append(DepthConvLayer.build_from_config(config_copy))

        config_copy = deepcopy(layer_config)
        key_del = ['dilation', 'groups', 'bias', 'has_shuffle', 'kernel_size', 'stride']
        for key in key_del:
            config_copy.pop(key)
        config_copy['use_bn'] = False
        config_copy['act_func'] = None
        candidates.append(IdentityLayer.build_from_config(config_copy))

        for i in range(3):
            config_copy = deepcopy(layer_config)
            config_copy['kernel_size'] = i*2+3
            config_copy['dilation'] = 1
            candidates.append(DepthConvLayer.build_from_config(config_copy))
        pool_type = ['avg', 'max']
        for i in range(len(pool_type)):
            config_copy = deepcopy(layer_config)
            key_del = ['dilation', 'groups', 'bias', 'has_shuffle']
            for key in key_del:
                config_copy.pop(key)
            config_copy['pool_type'] = pool_type[i]
            config_copy['kernel_size'] = 3
            config_copy['use_bn'] = False
            config_copy['act_func'] = None
            candidates.append(PoolingLayer.build_from_config(config_copy))
        ### NOTE
        return candidates

    if layer_name == 'PoolingLayer':
        # add a dilation conv
        config_copy = deepcopy(layer_config)
        config_copy['kernel_size'] = 3
        config_copy['stride'] = 1
        config_copy['dilation'] = 2
        config_copy['groups'] = 1
        config_copy['bias'] = False
        config_copy.pop('pool_type')
        candidates.append(DepthConvLayer.build_from_config(config_copy))
        # add a identity layer
        config_copy = deepcopy(layer_config)
        config_copy.pop('pool_type')
        config_copy.pop('kernel_size')
        config_copy.pop('stride')
        candidates.append(IdentityLayer.build_from_config(config_copy))
        # add 3 depth conv
        for i in range(3):
            config_copy = deepcopy(layer_config)
            config_copy['kernel_size'] = 2 * i + 3
            config_copy['stride'] = 1
            config_copy['dilation'] = 1
            config_copy['groups'] = 1
            config_copy['bias'] = False
            config_copy['has_shuffle'] = False
            config_copy.pop('pool_type')
            candidates.append(DepthConvLayer.build_from_config(config_copy))
        # add another pool layer
        pool_type = ['avg', 'max']
        for i in range(len(pool_type)):
            config_copy = deepcopy(layer_config)
            config_copy['pool_type'] = pool_type[i]
            candidates.append(PoolingLayer.build_from_config(config_copy))
        ### NOTE
        return candidates

    if layer_name == 'IdentityLayer':
        # add a dilation conv
        config_copy = deepcopy(layer_config)
        config_copy['kernel_size'] = 3
        config_copy['stride'] = 1
        config_copy['dilation'] = 2
        config_copy['groups'] = 1
        config_copy['bias'] = False
        config_copy['has_shuffle'] = False
        candidates.append(DepthConvLayer.build_from_config(config_copy))
        # add Identity
        config_copy = deepcopy(layer_config)
        candidates.append(IdentityLayer.build_from_config(config_copy))
        # add three depth conv
        for i in range(3):
            config_copy = deepcopy(layer_config)
            config_copy['kernel_size'] = 2 * i + 3
            config_copy['stride'] = 1
            config_copy['dilation'] = 1
            config_copy['groups'] = 1
            config_copy['bias'] = False
            config_copy['has_shuffle'] = False
            candidates.append(DepthConvLayer.build_from_config(config_copy))
        # add two pool layers
        pool_type = ['avg', 'max']
        for i in range(len(pool_type)):
            config_copy = deepcopy(layer_config)
            config_copy['pool_type'] = pool_type[i]
            config_copy['kernel_size'] = 3
            config_copy['stride'] = 1
            candidates.append(PoolingLayer.build_from_config(config_copy))
        return candidates

# which is the point of search, build_candidate_for_cell method is also for search
def set_layer_for_tree_cell():
    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return build_candidates_for_cell(layer_name, layer_config)



class BasicLayer(BasicUnit):

    def __init__(self, in_channels, out_channels,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(BasicLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ add modules """
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm2d(in_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        # activation
        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU(inplace=False)
            else:
                self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None
        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate, inplace=False)
        else:
            self.dropout = None

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_call(self, x):
        raise NotImplementedError

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.weight_call(x)
            elif op == 'bn':
                if self.bn is not None:
                    x = self.bn(x)
            elif op == 'act':
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError


class ConvLayer(BasicLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation
        # `kernel_size`, `stride`, `padding`, `dilation` can either be `int` or `tuple` of int
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                              padding=padding, dilation=self.dilation, groups=self.groups, bias=self.bias)

    def weight_call(self, x):
        x = self.conv(x)
        if self.has_shuffle and self.groups > 1:
            x = shuffle_layer(x, self.groups)
        return x

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        config = {
            'name': ConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
        }
        config.update(super(ConvLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)

    def get_flops(self, x):
        return count_conv_flop(self.conv, x), self.forward(x)


class DepthConvLayer(BasicLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        super(DepthConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation
        # `kernel_size`, `stride`, `padding`, `dilation` can either be `int` or `tuple` of int
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size, stride=self.stride,
                                    padding=padding, dilation=self.dilation, groups=in_channels, bias=False)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=self.groups, bias=self.bias)

    def weight_call(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        if self.has_shuffle and self.groups > 1:
            x = shuffle_layer(x, self.groups)
        return x

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            return '%dx%d_DilatedDepthConv' % (kernel_size[0], kernel_size[1])
        else:
            return '%dx%d_DepthConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        config = {
            'name': DepthConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
        }
        config.update(super(DepthConvLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)

    def get_flops(self, x):
        depth_flop = count_conv_flop(self.depth_conv, x)
        point_flop = count_conv_flop(self.point_conv, self.depth_conv(x))
        return depth_flop + point_flop, self.forward(x)


class PoolingLayer(BasicLayer):

    def __init__(self, in_channels, out_channels,
                 pool_type, kernel_size=2, stride=2,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(PoolingLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        if self.pool_type == 'avg':
            self.pool = nn.AvgPool2d(self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False)
        elif self.pool_type == 'max':
            self.pool = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError

    def weight_call(self, x):
        return self.pool(x)

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_%sPool' % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        config = {
            'name': PoolingLayer.__name__,
            'pool_type': self.pool_type,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
        }
        config.update(super(PoolingLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class IdentityLayer(BasicLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_call(self, x):
        return x

    @property
    def unit_str(self):
        return 'Identity'

    @property
    def config(self):
        config = {
            'name': IdentityLayer.__name__,
        }
        config.update(super(IdentityLayer, self).config)
        return config

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class LinearLayer(BasicUnit):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ add modules """
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm1d(in_features)
            else:
                self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = None
        # activation
        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU(inplace=False)
            else:
                self.activation = nn.ReLU(inplace=True)
        elif act_func == 'tanh':
            self.activation = nn.Tanh()
        elif act_func == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=False)
        else:
            self.dropout = None
        # linear
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                # dropout before weight operation
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.linear(x)
            elif op == 'bn':
                if self.bn is not None:
                    x = self.bn(x)
            elif op == 'act':
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x

    @property
    def unit_str(self):
        return '%dx%d_Linear' % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

    def get_flops(self, x):
        return self.linear.weight.numel(), self.forward(x)
