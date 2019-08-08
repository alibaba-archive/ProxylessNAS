""" Given a expdir, run the exp """

import numpy as np

import torch

from models.expdir_monitor import ExpdirMonitor

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    path = './Proxyless-NAS/Nets/cifar'
    expdir_monitor = ExpdirMonitor(path)
    expdir_monitor.train_samplenet()
    # expdir_monitor.train_supernet()
