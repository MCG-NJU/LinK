from typing import Any, Dict

import numpy as np
import torch
from torchpack import distributed as dist
from torchpack.callbacks.callback import Callback
import os


__all__ = ['MeanIoU']

kept_labels = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
]


class MeanIoU(Callback):

    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou',
                 run_dir: str) -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.run_dir = run_dir

    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        ## -1 for the ignored label [0]
        for i in range(self.num_classes-1):
            self.total_seen[i] += torch.sum(targets == i+1).item()
            self.total_correct[i] += torch.sum((targets == i+1)
                                               & (outputs == targets)).item()
            self.total_positive[i] += torch.sum(outputs == i+1).item()

    def _after_epoch(self) -> None:
        for i in range(self.num_classes-1):
            self.total_seen[i] = dist.allreduce(self.total_seen[i],
                                                reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i],
                                                   reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i],
                                                    reduction='sum')

        ious = []
        accs = []

        for i in range(self.num_classes-1):
            if self.total_seen[i] == 0:
                ious.append(1)
                accs.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                cur_acc = self.total_correct[i] / self.total_seen[i]                
                ious.append(cur_iou)
                accs.append(cur_acc)
        miou = np.mean(ious)
        macc = np.mean(accs)
        oacc = np.sum(self.total_correct) / np.sum(self.total_seen)

        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, miou * 100)        

        with open(os.path.join(self.run_dir, 'ious.txt'), 'a') as f:
            # uncomment to print iou of each category for SemanticKITTI
            # f.write("\n======== per class IoU ========\n")
            # print("\n======== per class IoU ========")
            # f.write(f"  {'label':<14}{'mIoU':>8}{'mAcc':>8}{'correct':>12}{'total':>12}\n")
            # print(f"  {'label':<14}{'mIoU':>8}{'mAcc':>8}{'correct':>12}{'total':>12}")
            # for i in range(self.num_classes-1):
            #     f.write(f"  {kept_labels[i]:<14}{ious[i]:>8.3%}{accs[i]:>8.3%}{int(self.total_correct[i]):>12,}{int(self.total_seen[i]):>12,}\n")
            #     print(f"  {kept_labels[i]:<14}{ious[i]:>8.3%}{accs[i]:>8.3%}{int(self.total_correct[i]):>12,}{int(self.total_seen[i]):>12,}")
            f.write(f"mIoU: {miou:.3%}\n")
            print(f"mIoU: {miou:.3%}")
            f.write(f"mAcc: {macc:.3%}\n")
            print(f"mAcc: {macc:.3%}")
            f.write(f"oAcc: {oacc:.3%}\n")
            print(f"oAcc: {oacc:.3%}")
