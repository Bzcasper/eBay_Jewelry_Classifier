import torch
from torch.optim.optimizer import Optimizer
import logging

"""
liger_optimizer.py

A custom optimizer called LigerOptimizer.
For demonstration:
- Similar to a basic SGD + weight decay approach.
- If needed, add momentum, adaptive learning rate logic.
- CPU-only. 
- Compatible with DeepSpeed initialize since we return standard parameter groups.

No placeholders: fully implemented logic.

You can adjust this logic as needed.
"""

class LigerOptimizer(Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0.01, momentum=0.9):
        """
        Initialize Liger optimizer with a simple momentum and weight decay logic.
        
        Args:
            params (iterable): Parameters to optimize.
            lr (float): Learning rate.
            weight_decay (float): Weight decay factor.
            momentum (float): Momentum factor.
        """
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, buffer=None)
        super(LigerOptimizer, self).__init__(params, defaults)
        logging.debug("LigerOptimizer initialized with lr={}, weight_decay={}, momentum={}".format(lr, weight_decay, momentum))

        for group in self.param_groups:
            group['momentum_buffer'] = {}

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        Uses a momentum-like update with weight decay.
        
        Closure is not typically needed here, but we allow it for compatibility.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Momentum buffer
                buf = group['momentum_buffer'].get(p)
                if buf is None:
                    buf = torch.clone(grad).detach()
                    group['momentum_buffer'][p] = buf
                else:
                    buf.mul_(momentum).add_(grad, alpha=1.0)

                p.add_(buf, alpha=-lr)

        return loss
