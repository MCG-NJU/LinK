from torch.nn.utils import clip_grad

from .hook import Hook
from torch.cuda import amp


class OptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip
        )

    def after_train_iter(self, trainer):

        trainer.optimizer.zero_grad()
        trainer.outputs["loss"].backward()
        if self.grad_clip is not None:
            self.clip_grads(trainer.model.parameters())
        trainer.optimizer.step()

        # -----------------

        # scaler = amp.GradScaler(enabled=True)
        # trainer.optimizer.zero_grad()
        # scaler.scale(trainer.outputs["loss"]).backward()
        # if self.grad_clip is not None:
        #     self.clip_grads(trainer.model.parameters())
        # scaler.step(trainer.optimizer)
        # scaler.update()
