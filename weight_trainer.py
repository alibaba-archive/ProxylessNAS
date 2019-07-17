#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class SampleNet(nn.Module):
    def __init__(self, num_classes, blocks,
                 dim_feature=1984,
                 ):
        super(SampleNet, self).__init__()
        self.index_candidate_block = [0]*22
        self.is_train_weight = True
        self.architecture_parameter = read_architecture_parameter()


        self._criterion = nn.CrossEntropyLoss().cuda()

        # self._ops = nn.ModuleList()
        self._ops = []
        self._blocks = blocks

        tmp = []
        input_conv_count = 0
        # the layer which is not to choose blocks.
        for b in blocks:
            if isinstance(b, nn.Module):
                tmp.append(b)
                input_conv_count += 1
            else:
                break
        self._input_conv = nn.Sequential(*tmp)
        self._input_conv_count = input_conv_count
        count_block = 0
        for b in blocks:
            if isinstance(b, list):
                self._ops.append(b)
                # self._ops.append(b[index_candidate_block[count_block]])
                count_block += 1
                input_conv_count += 1
        tmp = [] # to store output conv
        output_conv_count = 0
        for b in blocks[input_conv_count:]:
            if isinstance(b, nn.Module):
                tmp.append(b)
                output_conv_count += 1
            else:
                break
        self._output_conv = nn.Sequential(*tmp)
        # assert len(self.theta) == 22
        self.classifier = nn.Linear(dim_feature, num_classes)

    def forward(self, input):
        # when is_train_weight is True，choose different block to train weight each batch.
        if self.is_train_weight:
            softmax = nn.Softmax()
            index_candidate_block = []
            for i in self.architecture_parameter:
                output = softmax(torch.Tensor(i))
                index_candidate_block.append(output.multinomial(1).numpy().tolist()[0])
        else:
        # when is_train_weight is false，use index_candidate_block for inferencing
            index_candidate_block = self.index_candidate_block
        batch_size = input.size()[0]
        self.batch_size = batch_size
        data = self._input_conv(input)
        layer_of_block = 0
        for l_idx in range(self._input_conv_count, len(self._blocks)):
            block = self._blocks[l_idx]
            if isinstance(block, list):
                data = self._ops[layer_of_block][index_candidate_block[layer_of_block]](data)
                layer_of_block += 1
            else:
                break

        data = self._output_conv(data)
        # data = F.dropout(data, p=0.2)
        data = nn.functional.avg_pool2d(data, data.size()[2:])
        data = data.reshape((batch_size, -1))
        self.logits = self.classifier(data)
        return self.logits

def read_architecture_parameter():
    architecture_parameter = []
    with open('./architecture_parameter.txt','r') as f:
        for i in f.readlines():
            architecture_parameter.append([float(j) for j in i.strip().split(" ")])
    return architecture_parameter
