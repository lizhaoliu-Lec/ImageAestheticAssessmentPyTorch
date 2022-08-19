import torch.optim

from optimizer.factory import OptimizerFactory
from torch.optim import Optimizer
from torch.optim import AdamW
import math
@OptimizerFactory.register('SGD')
class SGD(torch.optim.SGD):
    ...


@OptimizerFactory.register('Adam')
class Adam(torch.optim.Adam):
    ...
@OptimizerFactory.register("AdamWarmup")
class AdamWarmup(Optimizer):
    # DOTA
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup = 0):
        assert  0.0 <= lr, "Invalid learning rate: {}".format(lr)
        assert  0.0 <= eps, "Invalid epsilon value: {}".format(eps)
           
        assert  0.0 <= betas[0] < 1.0,"Invalid beta parameter at index 0: {}".format(betas[0])
        assert  0.0 <= betas[1] < 1.0,"Invalid beta parameter at index 1: {}".format(betas[1])
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup)
        super().__init__(params, defaults)
        self.Currentlr=[0]*len(self.param_groups)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for i,group in enumerate(self.param_groups):

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']
                self.Currentlr[i]=scheduled_lr
                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss
    def get_lr(self):
        return self.Currentlr
# if __name__ =='main':
#     def run_adamwwarmup():
