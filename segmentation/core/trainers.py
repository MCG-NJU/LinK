from typing import Any, Callable, Dict

import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torchsparse import SparseTensor
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

__all__ = ['SemanticKITTITrainer']


class SemanticKITTITrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 amp_enabled: bool = False) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        _inputs = {}
        self.summary.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'])
        for key, value in feed_dict.items():
            if 'file_name' not in key:
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
        targets = _inputs['targets'].F.long().cuda(non_blocking=True)
        
        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(inputs)


            if outputs.requires_grad:
                if not isinstance(self.criterion, (list, tuple)):
                    loss = self.criterion(outputs, targets)
                else:
                    loss_ce = self.criterion[0](outputs, targets)

                    outputs_r = outputs.transpose(0,1).unsqueeze(dim=0).unsqueeze(dim=-1)
                    targets_r = targets.unsqueeze(dim=0).unsqueeze(dim=-1)
                    loss_lovasz = self.criterion[1](torch.nn.functional.softmax(outputs_r, dim=1), targets_r, ignore=0)

                    loss = loss_ce  + loss_lovasz
        
        if outputs.requires_grad:
            self.summary.add_scalar('loss', loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        else:
            # Calculate mIoU over all points.
            invs = _inputs['inverse_map']
            all_labels = _inputs['targets_mapped']
            _outputs = []
            _targets = []
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs.append(outputs_mapped)
                _targets.append(targets_mapped)
            outputs = torch.cat(_outputs, 0)
            targets = torch.cat(_targets, 0)

            # Calculate mIoU over sparse-quantized voxels, as an approximate result.

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass

