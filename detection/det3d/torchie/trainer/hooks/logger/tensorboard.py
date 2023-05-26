import os.path as osp

import torch
import numpy as np

from ...utils import master_only
from .base import LoggerHook


class TensorboardLoggerHook(LoggerHook):
    def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.log_dir = log_dir

    @master_only
    def before_run(self, trainer):
        if torch.__version__ >= "1.1":
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                )
        else:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError(
                    "Please install tensorboardX to use " "TensorboardLoggerHook."
                )

        if self.log_dir is None:
            self.log_dir = osp.join(trainer.work_dir, "tf_logs")
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, trainer):
        self.writer.add_scalar('lr', trainer.current_lr()[0], trainer.iter)
        watch_list = ['loss', 'hm_loss', 'loc_loss', 'num_positive']
        for var in watch_list:
            record = trainer.log_buffer.output[var]
            for i, r in enumerate(record):
                self.writer.add_scalar(f'{var}/{i}', r, trainer.iter)
            self.writer.add_scalar(f'{var}/avg', np.mean(record), trainer.iter)

    @master_only
    def after_run(self, trainer):
        self.writer.close()
