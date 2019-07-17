#!/usr/bin/python

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim
import torch.utils.data
import torchvision
import transforms as transforms

import torch.nn.functional as F


class SampleNet(nn.Module):
    def __init__(self, num_classes, blocks,
                 dim_feature=1984,
                 index_candidate_block=[0] * 22
                 ):
        super(SampleNet, self).__init__()

        self._criterion = nn.CrossEntropyLoss().cuda()

        self._ops = nn.ModuleList()
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
                self._ops.append(b[index_candidate_block[count_block]])
                count_block += 1
                input_conv_count += 1
        tmp = []  # to store output conv
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
        batch_size = input.size()[0]
        self.batch_size = batch_size
        data = self._input_conv(input)
        layer_of_block = 0
        for l_idx in range(self._input_conv_count, len(self._blocks)):
            block = self._blocks[l_idx]
            if isinstance(block, list):
                data = self._ops[layer_of_block](data)
                layer_of_block += 1
            else:
                break

        data = self._output_conv(data)
        # data = F.dropout(data, p=0.2)
        data = nn.functional.avg_pool2d(data, data.size()[2:])
        data = F.dropout(data, p=0.2)
        data = data.reshape((batch_size, -1))
        self.logits = self.classifier(data)
        return self.logits


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = './samplenet_checkpoints/' + args.project_name + '/' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './samplenet_checkpoints/' + args.project_name + '/' + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

### @ray.remote
class ValAcc(object):
    def __init__(self, model, index_candidate_block=[0]*22):
        self.index_candidate_block = index_candidate_block
        ### NOTE
        self.model = model

    def get_acc(self):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2473, 0.2434, 0.2610])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        valset = torchvision.datasets.CIFAR10(root='/data/volume1', train=False,
                                              download=True, transform=transform_val)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        criterion = nn.CrossEntropyLoss().cuda()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        end = time.time()

        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var).sum()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            # top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))

        return top1.avg


if __name__ == '__main__':
    main()
