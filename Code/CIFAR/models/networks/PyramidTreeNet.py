from models.networks.BasicBlockwiseTreeNet import *
from models.modules.layers import *


class PyramidTreeNet(BasicBlockwiseTreeNet):

    @property
    def config(self):
        config = {
            'name': PyramidTreeNet.__name__,
        }
        config.update(super(PyramidTreeNet, self).config)
        return config

    @staticmethod
    def build_from_config(config,  isSamplenet=False, indexs=None):
        ### NOTE
        blocks, classifier = BasicBlockwiseTreeNet.build_from_config(config, isSamplenet, indexs)
        return PyramidTreeNet(blocks, classifier, **config)

    @staticmethod
    def build_standard_net(cls, data_shape, n_classes, start_planes, alpha, block_per_group, total_groups,
                           downsample_type, bottleneck=4, ops_order='bn_act_weight', dropout_rate=0,
                           use_depth_sep_conv=False, groups_3x3=1, tree_node_config=None,
                           cell_drop_rate=0, cell_drop_scheme='uniform'):
        if tree_node_config is None:
            tree_node_config = {
                'use_avg': False,
                'skip_node_bn': False,
                'path_drop_rate': 0,
            }

        image_channel, image_size = data_shape[0:2]

        addrate = alpha / (block_per_group * total_groups)  # add pyramid_net

        # initial conv
        features_dim = start_planes
        if ops_order == 'weight_bn_act':
            init_conv_layer = ConvLayer(image_channel, features_dim, kernel_size=3, use_bn=True, act_func='relu',
                                        ops_order=ops_order)
        elif ops_order == 'act_weight_bn':
            init_conv_layer = ConvLayer(image_channel, features_dim, kernel_size=3, use_bn=True, act_func=None,
                                        ops_order=ops_order)
        elif ops_order == 'bn_act_weight':
            init_conv_layer = ConvLayer(image_channel, features_dim, kernel_size=3, use_bn=False, act_func=None,
                                        ops_order=ops_order)
        else:
            raise NotImplementedError
        init_bn_layer = IdentityLayer(features_dim, features_dim, use_bn=True, act_func=None, ops_order=ops_order)
        transition2blocks = TransitionBlock([init_conv_layer, init_bn_layer])

        blocks = [transition2blocks]

        # residual blocks
        total_blocks = total_groups * block_per_group
        planes = start_planes
        for group_idx in range(total_groups):
            for block_idx in range(block_per_group):
                if group_idx > 0 and block_idx == 0:
                    stride = 2
                    image_size //= 2
                else:
                    stride = 1
                """ prepare the residual block """
                planes += addrate
                # shortcut
                if stride == 1:
                    shortcut = IdentityLayer(features_dim, features_dim, use_bn=False, act_func=None,
                                             ops_order=ops_order)
                else:
                    if downsample_type == 'avg_pool':
                        shortcut = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=2, stride=2,
                                                use_bn=False, act_func=None, ops_order=ops_order)
                    elif downsample_type == 'max_pool':
                        shortcut = PoolingLayer(features_dim, features_dim, 'max', kernel_size=2, stride=2,
                                                use_bn=False, act_func=None, ops_order=ops_order)
                    else:
                        raise NotImplementedError
                # cell
                out_plane = int(round(planes))
                if out_plane % groups_3x3 != 0:
                    out_plane -= out_plane % groups_3x3  # may change to +=
                in_bottle = ConvLayer(features_dim, out_plane, kernel_size=1, use_bn=True, act_func=None,
                                      dropout_rate=dropout_rate, ops_order=ops_order)  # dropout in bottleneck layer
                cell = cls.build_tree_cell(use_depth_sep_conv, groups_3x3, out_plane, stride, tree_node_config,
                                           ops_order)

                out_bottle = ConvLayer(out_plane, out_plane * bottleneck, kernel_size=1, use_bn=True,
                                       act_func='relu', ops_order=ops_order)

                if cell_drop_scheme == 'linear':
                    _l = group_idx * block_per_group + block_idx + 1
                    current_cell_drop_rate = 2 * _l * cell_drop_rate / (total_blocks + 1)
                else:
                    current_cell_drop_rate = cell_drop_rate
                residual_block = ResidualTreeBlock(cell, in_bottle, out_bottle, shortcut,
                                                   final_bn=True, cell_drop_rate=current_cell_drop_rate)
                blocks.append(residual_block)
                features_dim = out_plane * bottleneck

        if ops_order == 'weight_bn_act':
            global_avg_pool = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=image_size, stride=image_size,
                                           use_bn=False, act_func=None, ops_order=ops_order)
        elif ops_order == 'act_weight_bn':
            global_avg_pool = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=image_size, stride=image_size,
                                           use_bn=False, act_func='relu', ops_order=ops_order)
        elif ops_order == 'bn_act_weight':
            global_avg_pool = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=image_size, stride=image_size,
                                           use_bn=True, act_func='relu', ops_order=ops_order)
        else:
            raise NotImplementedError
        transition2classes = TransitionBlock([global_avg_pool])
        blocks.append(transition2classes)
        classifier = LinearLayer(features_dim, n_classes, bias=True, use_bn=False, act_func=None, ops_order=ops_order)

        return PyramidTreeNet(blocks, classifier, ops_order)
