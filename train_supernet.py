import torch
from torch import nn
import torchvision.datasets as dset

import numpy as np
import logging
import argparse
import time
import os

from model import Trainer, FBNet
from data import get_ds
from blocks import get_blocks
from utils import _logger, _set_file


class Config(object):
    num_cls_used = 100
    init_theta = 1.0
    alpha = 0.2
    beta = 0.6
    speed_f = './speed.txt'
    w_lr = 0.1
    w_mom = 0.9
    w_wd = 1e-4
    t_lr = 0.01
    t_wd = 5e-4
    t_beta = (0.9, 0.999)
    init_temperature = 5.0
    temperature_decay = 0.956
    model_save_path = '/hdfs/input/v-chzen/code/fbnet_philly_cpu/'
    total_epoch = 90
    start_w_epoch = 2
    train_portion = 0.8

lr_scheduler_params = {
    'logger': _logger,
    'T_max': 400,
    'alpha': 1e-4,
    'warmup_step': 100,
    't_mul': 1.5,
    'lr_mul': 0.98,
}


if __name__ == '__main__':
    #subprocess.call(['pip','install','termcolor'])

    config = Config()

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
                                    and model parallel for classify net.")
    parser.add_argument('--batch-size', type=int, default=96,
                        help='training batch size of all devices.')
    parser.add_argument('--epochs', type=int, default=90,
                        help='number of training epochs.')
    parser.add_argument('--log-frequence', type=int, default=100,
                        help='log frequence, default is 100')
    parser.add_argument('--gpu', default='0,1,2,3',
                        help='gpu, default is 0')
    parser.add_argument('--load-model-path', type=str, default=None,
                        help='re_train, default is None')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='number of subprocesses used to fetch data, default is 4')
    parser.add_argument('--project-name', type=str, default='fbnet')    

    parser.add_argument('--alpha',type=float, default=0.2)

    parser.add_argument('--beta', type=float, default=0.6)
    
    parser.add_argument('--model-save-path', type=str, default='//blob')

    parser.add_argument('--speed_f', type=str, default='./speed.txt')
    
    parser.add_argument('--theta-result-path', type=str, default='//blob/theta-result' )

    parser.add_argument('--checkpoints-path', type=str, default='//blob/checkpoints')

    parser.add_argument('--dataDir')

    parser.add_argument('--modelDir')

    parser.add_argument('--logDir')

    parser.add_argument('--deviceId', type=int)

    args = parser.parse_args()
    print(args.theta_result_path)
    args.theta_result_path = '%s/%s' % (args.theta_result_path, args.project_name) 

    args.checkpoints_path = '%s/%s' % (args.checkpoints_path, args.project_name)

    args.model_save_path = '%s/%s/%s' % \
                           (args.model_save_path, time.strftime('%Y-%m-%d', time.localtime(time.time())), args.project_name)

    if not os.path.exists(args.model_save_path):
        _logger.warn("{} not exists, create it".format(args.model_save_path))
        os.makedirs(args.model_save_path)
    _set_file(args.model_save_path + 'log.log')

    import torchvision.transforms as transforms

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_data = dset.ImageFolder(root='/hdfs/input/v-chzen/data/imagenet/train100/train',
                            transform=train_transform)
    val_data = dset.ImageFolder(root='/hdfs/input/v-chzen/data/imagenet/train100/val',
                            transform=train_transform)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(config.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=16)

    val_queue = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size,
        pin_memory=True, num_workers=8)

    blocks = get_blocks(cifar10=False)
    model = FBNet(num_classes=config.num_cls_used if config.num_cls_used > 0 else 10,
                  blocks=blocks,
                  init_theta=config.init_theta,
                  alpha=args.alpha,
                  beta=args.beta,
                  speed_f=config.speed_f)

    trainer = Trainer(network=model,
                      w_lr=config.w_lr,
                      w_mom=config.w_mom,
                      w_wd=config.w_wd,
                      t_lr=config.t_lr,
                      t_wd=config.t_wd,
                      t_beta=config.t_beta,
                      init_temperature=config.init_temperature,
                      temperature_decay=config.temperature_decay,
                      logger=_logger,
                      gpus=args.gpu,
                      theta_result_path=args.theta_result_path,
                      checkpoints_path=args.checkpoints_path
                     )

    trainer.search(train_queue, val_queue,
                   total_epoch=config.total_epoch,
                   start_w_epoch=config.start_w_epoch,
                   log_frequence=args.log_frequence 
                  )
