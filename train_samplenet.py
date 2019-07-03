#!/usr/bin/python

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
#import torchvision.transforms as transforms
import transforms as transforms
import torchvision.datasets as datasets
from blocks_2 import get_blocks

import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

class SampleNet(nn.Module):
    def __init__(self, num_classes, blocks,
                 dim_feature=1984,
                 index_candidate_block = [0]*22
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
        data = data.reshape((batch_size, -1))
        self.logits = self.classifier(data)
        return self.logits


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=360, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float,
                        help='LR decay rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--isFBC', default=False)

parser.add_argument('--single_gpu', dest='single_gpu', action='store_true', default=False)

parser.add_argument('--multicrop', action='store_true', default=False)

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--project-name', type=str, default='seed_1')

parser.add_argument('--dataDir')

parser.add_argument('--modelDir')

parser.add_argument('--logDir')

parser.add_argument('--deviceId', type=int)

parser.add_argument('--gpus', type=str, default='0',
                        help='gpus, default is 0')
best_prec1 = 0

def read_theta():
    theta = []
    with open('./theta-result/split/_theta_epoch_91.txt','r') as f:
        for i in f.readlines():
            theta.append([float(j) for j in i.split(" ")])
    return theta


def main():
    global args, best_prec1
    args = parser.parse_args()

    print(args)

    # create model
    blocks = get_blocks(cifar10=False)
    theta = read_theta()
    torch.manual_seed(args.seed)
    softmax = nn.Softmax()
    index_candidate_block = []
    for i in theta:
        output = softmax(torch.Tensor(i))
        index_candidate_block.append(output.multinomial(1).numpy().tolist()[0])
 
    if args.isFBC == True:
        index_candidate_block = [0, 3, 8, 0, 0, 7, 6, 7, 3, 7, 6, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 3]
    model = SampleNet(num_classes=1000,
                      blocks=blocks,
                      index_candidate_block=index_candidate_block
                      )
    model = torch.nn.DataParallel(model, device_ids=[0] if args.single_gpu else None).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    for group in optimizer.param_groups:
        #print(group['lr'])
        group.setdefault('initial_lr', group['lr'])
    #optimizer_step = StepLR(optimizer, step_size=90, gamma=0.1)
    # optionally resume from a checkpoint
    file_name = '/hdfs/input/v-chzen/code/fbnet_philly_0415/samplenet_checkpoints/' + args.project_name + '/checkpoint.pth.tar'
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        checkpoint = torch.load(file_name)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(file_name))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join('/hdfs/input/v-chzen/data/imagenet_2/train/train', 'train')
    valdir = os.path.join('/hdfs/input/v-chzen/data/imagenet', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomLighting(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size//2, shuffle=False,
        num_workers=args.workers//2, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, multiple_crops=args.multicrop)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        #optimizer_step.step()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var).sum()

        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))
        prec1 = accuracy(output.data, target)
        top1.update(prec1.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, multiple_crops=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if multiple_crops:
        # XXX: Loss outputs are not valid (due to duplicated softmax)
        model = MultiCropEnsemble(model, 224, act=nn.functional.softmax)

    # switch to evaluate mode
    model.eval()

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
        #top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = './samplenet_checkpoints/'+ args.project_name + '/' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './samplenet_checkpoints/' + args.project_name + '/'+'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.gamma ** (epoch // 90))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    batch_size = target.size()[0]
    pred = torch.argmax(output, dim=1)
    acc = torch.sum(pred == target).float() / batch_size
    return acc


if __name__ == '__main__':
    main()
