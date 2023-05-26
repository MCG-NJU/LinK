import argparse
import random
import shutil
import sys
import os
import time
import pathlib
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
import torchpack
from torchpack.callbacks import InferenceRunner, MaxSaver, Saver
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

import torchsparse
sys.path.append('..')
from core import builder
from core.callbacks import MeanIoU
from core.trainers import SemanticKITTITrainer

ROOT = '.'

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    args, opts = parser.parse_known_args()
    
    
    configs.load(osp.join(ROOT, args.config), recursive=True)
    configs.update(opts)

    if configs.distributed:
        dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        cur_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
        if configs.model.name in ['linkunet', 'linkencoder']:
            args.run_dir = os.path.join(ROOT, args.run_dir, f'{configs.model.base_op}[{configs.model.r}x{configs.model.s}_g{configs.model.groups}]', cur_time)
        else:
            args.run_dir = os.path.join(ROOT, args.run_dir, cur_time)
        set_run_dir(args.run_dir)
        

    # save runtime code for reproduction
    backup_dir = os.path.join(args.run_dir, 'backup')
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(backup_dir):
            saveRuntimeCode(backup_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2 ** 32 - 1)

    seed = configs.train.seed + dist.rank(
    ) * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = builder.make_dataset()

    dataflow = {}
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size if split=='train' else 1,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)
    
    resume = False
    if not resume:
        model = builder.make_model().cuda()
        if configs.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[dist.local_rank()], find_unused_parameters=True)
        
        optimizer = builder.make_optimizer(model)
        criterion = builder.make_criterion()
        scheduler = builder.make_scheduler(optimizer)
        
    else:
        model = builder.make_model().cuda()
        optimizer = builder.make_optimizer(model)
        criterion = builder.make_criterion()
        scheduler = builder.make_scheduler(optimizer)

        model, optimizer, scheduler = resume_model(configs.resume_path, model, optimizer, scheduler)



    trainer = SemanticKITTITrainer(model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   num_workers=configs.workers_per_gpu,
                                   seed=seed,
                                   amp_enabled=configs.amp_enabled)

    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.num_epochs,
        callbacks=[
            InferenceRunner(
                dataflow[split],
                callbacks=[
                    MeanIoU(name=f'iou/{split}',
                            num_classes=configs.data.num_classes,
                            ignore_label=configs.data.ignore_label,
                            run_dir=args.run_dir)
                ],
            ) for split in ['val']
        ] + [
            MaxSaver('iou/val'),
            Saver(max_to_keep=4),
        ])


def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    with open(osp.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    spvnas_dir = pathlib.Path(__file__).parent.resolve()

    shutil.copytree(spvnas_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def resume_model(name: str, model, optimizer, scheduler):
    init = torch.load(f'{name}/max-iou-val.pt',
                        map_location='cuda:%d' % dist.local_rank()
                        if torch.cuda.is_available() else 'cpu')
    state_dict = init['model']
    optim = init['optimizer']
    sched = init['scheduler']
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
            name = k[7:] # remove '.module'
            new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(optim)
    scheduler.load_state_dict(sched)
    
    logger.info(f'checkpoint resumed~~')
    return model, optimizer, scheduler

if __name__ == '__main__':
    main()
