from __future__ import division
import math
import random

from models.utils import *
from models.modules.layers import set_layer_from_config
from models.modules.tree_nodes import set_tree_node_from_config
from models.modules.tree_nodes import Count


def set_block_from_config(config, isSamplenent=False, index=None):
    name2block = {
        TransitionBlock.__name__: TransitionBlock,
        ResidualTreeBlock.__name__: ResidualTreeBlock,
    }

    block_name = config.pop('name')
    block = name2block[block_name]
    if block_name == 'TransitionBlock':
        return block.build_from_config(config)
    else:
        return block.build_from_config(config, isSamplenent, index)



class BasicBlockwiseTreeNet(BasicUnit):

    def __init__(self, blocks, classifier, ops_order):
        super(BasicBlockwiseTreeNet, self).__init__()

        self.blocks = nn.ModuleList(blocks)
        self.classifier = classifier

        self.ops_order = ops_order

        self.candidate_index = []

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    def set_candidate_for_blocks(self, candidate_index):
        for i in range(1,len(self.blocks)-1):
            self.blocks[i].set_candidate_for_block(candidate_index[i-1])

    @property
    def unit_str(self):
        raise ValueError('not needed')

    @property
    def config(self):
        return {
            'ops_order': self.ops_order,
            'blocks': [
                block.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config, isSamplenet=False, indexs=None):
        if 'name' in config:
            config.pop('name')
        blocks = []
        ### NOTE
        i = 0
        for block_config in config.pop('blocks'):
            if isSamplenet and (i != 0 and i != 55):
                block = set_block_from_config(block_config, isSamplenet, indexs.pop(0))
            else:
                block = set_block_from_config(block_config)
            blocks.append(block)
            i += 1
        classifier = set_layer_from_config(config.pop('classifier'))
        return blocks, classifier

    def get_flops(self, x):
        flop = 0
        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop
        x = x.view(x.size(0), -1)  # flatten
        delta_flop, x = self.classifier.get_flops(x)
        return flop + delta_flop, x

    def init_model(self, model_init, init_div_groups=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def build_standard_net(**kwargs):
        raise NotImplementedError

    def weight_parameters(self):
        return self.parameters()


class TransitionBlock(BasicUnit):

    def __init__(self, layers):
        super(TransitionBlock, self).__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def unit_str(self):
        raise ValueError('not needed')

    @property
    def config(self):
        return {
            'name': TransitionBlock.__name__,
            'layers': [
                layer.config for layer in self.layers
            ]
        }

    @staticmethod
    def build_from_config(config):
        layers = []
        for layer_config in config.get('layers'):
            layer = set_layer_from_config(layer_config)
            layers.append(layer)
        block = TransitionBlock(layers)
        return block

    def get_flops(self, x):
        flop = 0
        for layer in self.layers:
            delta_flop, x = layer.get_flops(x)
            flop += delta_flop
        return flop, x


class ResidualTreeBlock(BasicUnit):
    def __init__(self, cell, in_bottle, out_bottle, shortcut, final_bn=True, cell_drop_rate=0):
        super(ResidualTreeBlock, self).__init__()

        self.cell = cell
        self.in_bottle = in_bottle
        self.out_bottle = out_bottle
        self.shortcut = shortcut

        if final_bn:
            self.final_bn = nn.BatchNorm2d(self.out_channels)
        else:
            self.final_bn = None

        self.cell_drop_rate = cell_drop_rate

    def set_candidate_for_block(self, candidate_index):
        self.cell.set_candidate_for_nodes(candidate_index, 0)
    @property
    def out_channels(self):
        if self.out_bottle is None:
            out_channels = self.cell.out_channels
        else:
            out_channels = self.out_bottle.out_channels
        return out_channels

    def cell_normal_forward(self, x):
        if self.in_bottle is not None:
            x = self.in_bottle(x)

        x = self.cell(x)

        if self.out_bottle is not None:
            x = self.out_bottle(x)
        if self.final_bn:
            x = self.final_bn(x)
        return x

    def forward(self, x):
        _x = self.shortcut(x)
        batch_size = _x.size()[0]
        feature_map = _x.size()[2:4]

        if self.cell_drop_rate > 0:
            if self.training:
                # train
                p = random.uniform(0, 1)
                drop_flag = p < self.cell_drop_rate
                if drop_flag:
                    padding = torch.zeros(batch_size, self.out_channels, feature_map[0], feature_map[1])
                    if _x.is_cuda:
                        padding = padding.cuda()
                    x = torch.autograd.Variable(padding, requires_grad=False)
                else:
                    x = self.cell_normal_forward(x) / (1 - self.cell_drop_rate)
            else:
                # test
                x = self.cell_normal_forward(x)
        else:
            x = self.cell_normal_forward(x)

        residual_channel = x.size()[1]
        shortcut_channel = _x.size()[1]
        if residual_channel != shortcut_channel:
            padding = torch.zeros(batch_size, residual_channel - shortcut_channel, feature_map[0], feature_map[1])
            if _x.is_cuda:
                padding = padding.cuda()
            padding = torch.autograd.Variable(padding, requires_grad=False)
            _x = torch.cat((_x, padding), 1)

        return _x + x

    @property
    def unit_str(self):
        raise ValueError('not needed')

    @property
    def config(self):
        return {
            'name': ResidualTreeBlock.__name__,
            'cell_drop_rate': self.cell_drop_rate,
            'final_bn': self.final_bn is not None,
            'shortcut': self.shortcut.config,
            'in_bottle': self.in_bottle.config,
            'out_bottle': self.out_bottle.config,
            'cell': self.cell.config,
        }

    @staticmethod
    def build_from_config(config, isSamplenet=False, index=None):
        if config.get('in_bottle'):
            in_bottle = set_layer_from_config(config.get('in_bottle'))
        else:
            in_bottle = None

        if config.get('out_bottle'):
            out_bottle = set_layer_from_config(config.get('out_bottle'))
        else:
            out_bottle = None
        shortcut = set_layer_from_config(config.get('shortcut'))
        ### NOTE
        count = Count()

        cell = set_tree_node_from_config(config.get('cell'), count, isSamplenet, index)

        final_bn = config.get('final_bn')
        cell_drop_rate = config.get('cell_drop_rate')

        return ResidualTreeBlock(cell, in_bottle, out_bottle, shortcut, final_bn, cell_drop_rate)

    def get_flops(self, x):
        flop, _x = self.shortcut.get_flops(x)
        batch_size = _x.size()[0]
        feature_map = _x.size()[2:4]

        if self.in_bottle is not None:
            delta_flop, x = self.in_bottle.get_flops(x)
            flop += delta_flop

        delta_flop, x = self.cell.get_flops(x)
        flop += delta_flop

        if self.out_bottle is not None:
            delta_flop, x = self.out_bottle.get_flops(x)
            flop += delta_flop

        residual_channel = x.size()[1]
        shortcut_channel = _x.size()[1]
        if residual_channel != shortcut_channel:
            padding = torch.zeros(batch_size, residual_channel - shortcut_channel, feature_map[0], feature_map[1])
            if _x.is_cuda:
                padding = padding.cuda()
            padding = torch.autograd.Variable(padding, requires_grad=False)
            _x = torch.cat((_x, padding), 1)

        return flop, _x + x
