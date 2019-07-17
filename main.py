# import ray
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision

from weight_trainer import SampleNet
from blocks_cifar import get_blocks

import transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch ProxylessNAS Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
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
parser.add_argument('--single_gpu', dest='single_gpu', action='store_true', default=False)

parser.add_argument('--seed', type=int, default=1)

def main():
    ### ray.init()
    global args, best_prec1
    args = parser.parse_args()

    print(args)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2473, 0.2434, 0.2610])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    blocks = get_blocks(cifar10=True)
    model = SampleNet(num_classes=10, blocks=blocks)
    ### model = torch.nn.DataParallel(model, device_ids=[0] if args.single_gpu else None).cuda()
    model.architecture_parameters = read_architecture_parameters()
    architecture_parameters = model.architecture_parameters
    ### NOTE
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train weight for one epoch
        model.is_train_weight = True
        print('train_weight_parameters')
        # train_weight_parameters(train_loader, optimizer, model, criterion, epoch)

        # Configure 20 workers and get acc by interacting with the environment
        index_candidate_blocks = [get_index_candidate_blocks(architecture_parameters) for _ in range(20)]
        model.is_train_weight = False
        models = []
        for i in range(20):
            model.index_candidate_block = index_candidate_blocks[i]
            models.append(model)
        ### val_accs = [ValAcc.remote(val_loader, models[i]) for i in range(20)]
        val_accs = [ValAcc(models[i]) for i in range(20)]

        ### acc_results = ray.get([v.get_acc.remote() for v in val_accs])
        acc_results = [v.get_acc() for v in val_accs]
        print('train_architecture_parameters')
        architecture_parameters = train_architecture_parameters(architecture_parameters, index_candidate_blocks, acc_results)

def read_architecture_parameters():
    architecture_parameters = []
    with open('./architecture_parameter.txt','r') as f:
        for i in f.readlines():
            architecture_parameters.append([float(j) for j in i.strip().split(" ")])
    return architecture_parameters

def get_index_candidate_blocks(architecture_parameters):
    softmax = nn.Softmax()
    index_candidate_block = []
    for i in architecture_parameters:
        output = softmax(torch.Tensor(i))
        index_candidate_block.append(output.multinomial(1).numpy().tolist()[0])
    return index_candidate_block

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 90 epochs"""
    lr = args.lr * (args.gamma ** (epoch // 90))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_architecture_parameters(architecture_parameters, index_candidate_blocks, acc_results):
    M = len(acc_results)
    for layer in range(len(architecture_parameters)):
        sum_layer = sum([math.exp(i) for i in architecture_parameters[layer]])
        for op in range(len(architecture_parameters[layer])):
            grad = 0
            for m in range(M):
                if op == index_candidate_blocks[m][layer]:

                    grad += 1/M * acc_results[m] * (1 - math.exp(architecture_parameters[layer][op])/sum_layer)
                else:
                    grad += 1/M * acc_results[m] * (-1 * math.exp(architecture_parameters[layer][op])/sum_layer)
            architecture_parameters[layer][op] -= grad

    return architecture_parameters

def train_weight_parameters(train_loader, optimizer, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    softmax = nn.Softmax()
    for i, (input, target) in enumerate(train_loader):
        # updates different path each batch
        index_candidate_block = []
        for j in model.architecture_parameters:
            output = softmax(torch.Tensor(j))
            index_candidate_block.append(output.multinomial(1).numpy().tolist()[0])
        model.index_candidate_block = index_candidate_block

        # measure data loading time
        data_time.update(time.time() - end)

        ### target = target.cuda(async=True)
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
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

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

def accuracy(output, target):
    batch_size = target.size()[0]
    pred = torch.argmax(output, dim=1)
    acc = torch.sum(pred == target).float() / batch_size
    return acc

### @ray.remote
class ValAcc(object):
    def __init__(self, model):
        self.model = model

    def get_acc(self):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2473, 0.2434, 0.2610])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform_val)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.CrossEntropyLoss().cuda()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        end = time.time()

        for i, (input, target) in enumerate(val_loader):

            ### target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = self.model(input_var)
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

