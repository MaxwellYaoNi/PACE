"""
Implementation of SharableDropPath:
If your network contains modules like Dropout that apply noise, it's better to implement a sharable version.
"""

import timm
import torch

class SharableDropPath(timm.layers.DropPath):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True, duplicate=2, **kwargs):
        super(SharableDropPath, self).__init__(drop_prob, scale_by_keep)
        self.duplicate = duplicate

    def set_duplicate(self, duplicate):
        self.duplicate = duplicate

    def forward(self, x):
        # below code is the same as `timm.layers.drop_path`
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        ###<add> we add the code below to ensure the same sample shares the same Bernoulli noise.
        if self.duplicate > 1:
            bs = x.shape[0] // self.duplicate
            random_tensor = torch.cat([random_tensor[:bs]] * self.duplicate, dim=0)
        random_tensor = random_tensor / (random_tensor.mean() + 1e-12)
        random_tensor = random_tensor.detach()
        ###</add>
        return x * random_tensor

def ensure_sharable_drop_path(model, duplicate=2):
    for name, l in model.named_modules():
        if isinstance(l, timm.layers.DropPath):
            parent_layer = model
            tokens = name.strip().split('.')
            for t in tokens[:-1]:
                parent_layer = parent_layer[int(t)] if t.isnumeric() else getattr(parent_layer, t)
            drop_path = getattr(parent_layer, tokens[-1])
            setattr(parent_layer, tokens[-1], SharableDropPath(drop_path.drop_prob, drop_path.scale_by_keep, duplicate))


def set_duplicate(model, duplicate):
    for _, l in model.named_modules():
        if isinstance(l, SharableDropPath):
            l.set_duplicate(duplicate)