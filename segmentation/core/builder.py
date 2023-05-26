from typing import Callable

import torch
import torch.optim
from torch import nn
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler
from .lovasz_losses import lovasz_softmax

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler'
]


def make_dataset() -> Dataset:
    if configs.dataset.name == 'semantic_kitti':
        from core.datasets import SemanticKITTI
        dataset = SemanticKITTI(root=configs.dataset.root,
                                num_points=configs.dataset.num_points,
                                voxel_size=configs.dataset.voxel_size,
                                )
    elif configs.dataset.name == 'nuscenes':
        from core.datasets import nuScenes
        dataset = nuScenes(root=configs.dataset.root,
                                num_points=configs.dataset.num_points,
                                voxel_size=configs.dataset.voxel_size,
                                )
    else:
        raise NotImplementedError(configs.dataset.name)
    return dataset


def make_model() -> nn.Module:
    if 'cr' in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0
    if configs.model.name == 'minkunet':
        from core.models.semantic_kitti import MinkUNet
        model = MinkUNet(num_classes=configs.data.num_classes, cr=cr)
    elif configs.model.name == 'linkunet':
        from core.models.semantic_kitti import ELKUNet
        baseop = configs.model.base_op
        model = ELKUNet(num_classes=configs.data.num_classes, cr=cr, r=configs.model.r, s=configs.model.s, groups=configs.model.groups, baseop=baseop)
    elif configs.model.name == 'linkencoder':
        from core.models.semantic_kitti import ELKEncoder
        baseop = configs.model.base_op
        model = ELKEncoder(num_classes=configs.data.num_classes, cr=cr, r=configs.model.r, s=configs.model.s, groups=configs.model.groups,baseop=baseop)
    elif configs.model.name == 'spvcnn':
        from core.models.semantic_kitti import SPVCNN
        model = SPVCNN(num_classes=configs.data.num_classes,
                       cr=cr,
                       pres=configs.dataset.voxel_size,
                       vres=configs.dataset.voxel_size)
    else:
        raise NotImplementedError(configs.model.name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(
            ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'lovasz_softmax':
        criterion_ce = nn.CrossEntropyLoss(
            ignore_index=configs.criterion.ignore_index)
        criterion_lovasz = lovasz_softmax
        criterion = [criterion_ce, criterion_lovasz]
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    lr = configs.optimizer.lr
    params = model.parameters()
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=lr,
                                    momentum=configs.optimizer.momentum,
                                    weight_decay=configs.optimizer.weight_decay,
                                    nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.num_epochs)
    elif configs.scheduler.name == 'cosine_warmup':
        from functools import partial

        from core.schedulers import cosine_schedule_with_warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(cosine_schedule_with_warmup,
                              num_epochs=configs.num_epochs,
                              batch_size=configs.batch_size,
                              dataset_size=configs.data.training_size))
    elif configs.scheduler.name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.1,
                                                               verbose=True,
                                                               patience=2,
                                                               min_lr=1e-6)
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
