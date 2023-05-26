import argparse
import sys

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
from torchsparse import SparseTensor
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import Callbacks, SaverRestore
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from tqdm import tqdm

from core import builder
from core.callbacks import MeanIoU
from core.trainers import SemanticKITTITrainer

import json
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('config', metavar='FILE',help='config file')
parser.add_argument('--run-dir', metavar='DIR', help='run directory')
parser.add_argument('--name', type=str, default='linkunet', help='model name')
parser.add_argument('--load_path', type=str, help='ckpt path')
args, opts = parser.parse_known_args()


def main() -> None:
    eval_on_dataset = 'test'

    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    
    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

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
            batch_size=configs.batch_size if split == 'train' else 1,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    model = find_model(args.name, configs, dist.local_rank())
    model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[dist.local_rank()], find_unused_parameters=True)

    model.eval()

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = SemanticKITTITrainer(model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   num_workers=8,
                                   seed=configs.train.seed)
    callbacks = Callbacks([
        SaverRestore(),
        MeanIoU(configs.data.num_classes, configs.data.ignore_label, 
                run_dir=args.run_dir)
    ])
    callbacks._set_trainer(trainer)
    trainer.callbacks = callbacks
    trainer.dataflow = dataflow[eval_on_dataset]


    trainer.before_train()
    trainer.before_epoch()

    model.eval()
    with torch.no_grad():
      for feed_dict in tqdm(dataflow[eval_on_dataset], desc='eval'):
          torch.cuda.empty_cache()
          _inputs = {}
          for key, value in feed_dict.items():
              if 'name' not in key:
                  if value.C.shape[1]==4:
                      _inputs[key] = value.cuda()
                  else:
                      tmp = value
                      tmp_C = tmp.C[:,:4]
                      tmp_F = tmp.F
                      _inputs[key] = SparseTensor(tmp_F, tmp_C).cuda()
              else:
                  path = value
                    
          inputs = _inputs['lidar']
  
          outputs = model(inputs)
  
          # Calculate mIoU over all points.
          invs = _inputs['inverse_map']
          _outputs = []
          for idx in range(invs.C[:, -1].max() + 1):
              cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
              cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
              outputs_mapped = outputs[cur_scene_pts][cur_inv] # .argmax(1)
              _outputs.append(outputs_mapped)
          outputs = torch.stack(_outputs, dim=0).sum(dim=0).argmax(1)
          saving = True
          if saving:
              sequence = path[0].split('/')[-3]
              fid = path[0].split('/')[-1].replace('bin', 'label')
              predictions_dir = os.path.join('submission/sequences', sequence, 'predictions')
              os.makedirs(predictions_dir, exist_ok=True)
              label_name = fid
              predictions = outputs.cpu().numpy()

              saveBinLabel(os.path.join(predictions_dir, fid), predictions)


def find_model(model_name: str, configs, local_rank):
    if 'cr' in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0
    if model_name == 'linkunet':
        from core.models.semantic_kitti import ELKUNet
        model = ELKUNet(num_classes=configs.data.num_classes,
                        cr=cr,
                        baseop=configs.model.base_op,
                        r=configs.model.r, 
                        s=configs.model.s,
                        groups=configs.model.groups)
    elif model_name == 'linkencoder':
        from core.models.semantic_kitti import ELkEncoder
        model = ELKEncoder(num_classes=configs.data.num_classes, 
                        cr=cr, 
                        baseop=configs.model.base_op,
                        r=configs.model.r, 
                        s=configs.model.s,
                        groups=configs.model.groups)
    elif model_name == 'mink':
        from core.models.semantic_kitti import MinkUNet
        model = MinkUNet(num_classes=configs.data.num_classes, cr=cr)
    elif model_name == 'spvcnn':
        from core.models.semantic_kitti import SPVCNN
        model = SPVCNN(num_classes=configs.data.num_classes,
                        cr=cr,
                        pres=configs.dataset.voxel_size,
                        vres=configs.dataset.voxel_size)
    else:
        raise NotImplementedError(model_name)

    init = torch.load(args.load_path, map_location='cuda:{}'.format(local_rank))
    state_dict = init['model']

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v  # remove `module.`

    # load params
    model.load_state_dict(new_state_dict)

    return model

def saveBinLabel(labelFilename, predictions):
    if os.path.exists(labelFilename):
        logger.info("===> {} already exist, skip.".format(labelFilename.split("/")[-1]))
        return
    predictions.astype(np.uint32)
    logger.info(predictions.shape)
    logger.info(predictions.dtype)
                                                                    
    def save_bin(binName, pred):
        with open(binName, 'wb+') as binfile :
            for i in range(pred.shape[0]):
                class_label = pred[i]
                class_label = int(class_label)
                                                                                                                                                                                                                       
                content = class_label.to_bytes(4, byteorder = 'little')     # 4字节，小端
                # logging.info({class_label:content})
                # byt4 = (1).to_bytes(4, byteorder = 'big')
                # print(byt4)
                binfile.write(content )
                                                                                                                                                                                                                                                                                                                # label_map_inv
    label_map_inv = {
        0: 0,  # "unlabeled", and others ignored
        1: 10,   # "car"
        2: 11,     # "bicycle"
        3: 15,    # "motorcycle"
        4: 18,   # "truck"
        5: 20,     # "other-vehicle"
        6: 30,     # "person"
        7: 31,     # "bicyclist"
        8: 32,     # "motorcyclist"
        9: 40,     # "road"
        10: 44,    # "parking"
        11: 48,    # "sidewalk"
        12: 49,    # "other-ground"
        13: 50,    # "building"
        14: 51,    # "fence"
        15: 70,    # "vegetation"
        16: 71,    # "trunk"
        17: 72,    # "terrain"
        18: 80,    # "pole"
        19: 81    # "traffic-sign"
    }
    # make lookup table for mapping
    maxkey = max(label_map_inv.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(label_map_inv.keys())] = list(label_map_inv.values())
    # logger.info("remap_lut:{}".format(remap_lut))
    predictions_inv = remap_lut[predictions]
    # upper_half = np.zeros((predictions.shape), dtype=np.uint32)
    # save_bin(labelFilename, predictions)
    predictions_inv.astype(np.uint32)
    logger.info("predictions:{}".format(set(predictions)))
    logger.info("predictions_inv:{}".format(set(predictions_inv)))
    predictions_inv.tofile(labelFilename)
    logger.info('===> Save label file:{}'.format(labelFilename))
    
    logger.info("*"*80)
    
    
    def loadBinLabel( labelFilename):
        logger.info("===> Loading {}".format(labelFilename))
        label = np.fromfile(labelFilename, dtype=np.uint32)
        label = label.reshape((-1))
        logger.info("===> [Load]label:{}".format(set(label)))
        sem_label = label & 0xFFFF  # 获取低16位信息
        logger.info("===> [Load]sem_label :{}".format(set(sem_label)))
        # label_map
        remapdict = {
            0 : 0,     # "unlabeled"
            1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 1,     # "car"
            11: 2,     # "bicycle"
            13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 3,     # "motorcycle"
            16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 4,     # "truck"
            20: 5,     # "other-vehicle"
            30: 6,     # "person"
            31: 7,     # "bicyclist"
            32: 8,     # "motorcyclist"
            40: 9,     # "road"
            44: 10,    # "parking"
            48: 11,    # "sidewalk"
            49: 12,    # "other-ground"
            50: 13,    # "building"
            51: 14,    # "fence"
            52: 0,      # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 9,     # "lane-marking" to "road" ---------------------------------mapped
            70: 15,    # "vegetation"
            71: 16,    # "trun" 
            72: 17,    # "terrain"
            80: 18,    # "pole"
            81: 19,    # "traffic-sign"
            99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
            252: 1,    # "moving-car" to "car" ------------------------------------mapped
            253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 6,    # "moving-person" to "person" ------------------------------mapped
            255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 4,    # "moving-truck" to "truck" --------------------------------mapped
            259: 5    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        # make lookup table for mapping
        maxkey = max(remapdict.keys())
        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        # logger.info(remap_lut)
        sem_label_mapped = remap_lut[sem_label]  # do the remapping of semantics
        logger.info("===> [Load]sem_label_mapped :{}".format(set(sem_label_mapped)))
        sem_label_mapped-=1
        return sem_label_mapped
        pass

if __name__ == '__main__':
    main()
