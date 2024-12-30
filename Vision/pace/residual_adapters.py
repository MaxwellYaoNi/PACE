"""
Implementation for Residual Adapters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualAdapter(nn.Module):
    def __init__(self, base_module, merge_training=True, merge_validation=True):
        super(ResidualAdapter, self).__init__()
        self.base_module = base_module
        self.merge_training = merge_training
        self.merge_validation = merge_validation

    def forward_base(self, x: torch.Tensor, **kwargs):
        return self.base_module(x, **kwargs)

    def forward_adapter(self, x: torch.Tensor, **kwargs):
        raise NotImplementedError

    def forward_merge(self, x: torch.Tensor, **kwargs):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, **kwargs):
        if (self.training and self.merge_training) or (not self.training and self.merge_validation):
            return self.forward_merge(x, **kwargs)
        else:
            return self.forward_base(x, **kwargs) + self.forward_adapter(x, **kwargs)

class LoRAmul_VPTadd_Linear(ResidualAdapter):
    def __init__(self, base_module:nn.Linear, rank=10, merge_training=True, merge_validation=True):
        super(LoRAmul_VPTadd_Linear, self).__init__(base_module, merge_training, merge_validation)
        out_features, in_features = base_module.weight.shape
        self.delta_W_down = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_W_up   = nn.Parameter(torch.zeros(rank, in_features))
        self.delta_b      = nn.Parameter(torch.zeros(out_features))
        self.delta_prompt_down = nn.Parameter(torch.zeros(in_features, rank))
        self.delta_prompt_up   = nn.Parameter(torch.zeros(rank, 1))

        nn.init.xavier_uniform_(self.delta_W_up)
        nn.init.xavier_uniform_(self.delta_prompt_up)

    def forward_adapter(self, x:torch.Tensor, **kwargs):
        W = self.base_module.weight * (self.delta_W_down @ self.delta_W_up)
        b = (self.base_module.weight @ (self.delta_prompt_down @ self.delta_prompt_up)).squeeze()
        if self.base_module.bias is not None: b = b + self.base_module.bias * self.delta_b
        return F.linear(x, W, b)

    def forward_merge(self, x: torch.Tensor, **kwargs):
        W = self.base_module.weight * (1 + self.delta_W_down @ self.delta_W_up)
        b = (self.base_module.weight @ (self.delta_prompt_down @ self.delta_prompt_up)).squeeze()
        if self.base_module.bias is not None: b = b + self.base_module.bias * (1 + self.delta_b)
        return F.linear(x, W, b)

class LoRAadd_Linear(ResidualAdapter):
    def __init__(self, base_module:nn.Linear, rank=4, merge_training=True, merge_validation=True):
        super(LoRAadd_Linear, self).__init__(base_module, merge_training, merge_validation)
        out_features, in_features = base_module.weight.shape
        self.delta_W_down = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_W_up   = nn.Parameter(torch.zeros(rank, in_features))
        self.delta_b      = nn.Parameter(torch.zeros(out_features))

        nn.init.xavier_uniform_(self.delta_W_up)

    def forward_adapter(self, x:torch.Tensor, **kwargs):
        return F.linear(F.linear(x, self.delta_W_up, None), self.delta_W_down, self.delta_b)

    def forward_merge(self, x: torch.Tensor, **kwargs):
        W = self.base_module.weight + self.delta_W_down @ self.delta_W_up
        b = self.delta_b
        if self.base_module.bias is not None: b = b + self.base_module.bias
        return F.linear(x, W, b)

class VPTadd_Linear(ResidualAdapter):
    def __init__(self, base_module:nn.Linear, rank=4, merge_training=True, merge_validation=True):
        super(VPTadd_Linear, self).__init__(base_module, merge_training, merge_validation)
        out_features, in_features = base_module.weight.shape
        self.delta_prompt_down = nn.Parameter(torch.zeros(in_features, rank))
        self.delta_prompt_up = nn.Parameter(torch.zeros(rank, 1))

        nn.init.xavier_uniform_(self.delta_prompt_up)

    def forward_adapter(self, x:torch.Tensor, **kwargs):
        return (self.base_module.weight @ (self.delta_prompt_down @ self.delta_prompt_up)).view(1, -1).expand(*x.shape[:-1], -1)

    def forward_merge(self, x: torch.Tensor, **kwargs):
        W = self.base_module.weight
        b = (self.base_module.weight @ (self.delta_prompt_down @ self.delta_prompt_up)).squeeze()
        if self.base_module.bias is not None: b = b + self.base_module.bias
        return F.linear(x, W, b)


class HeadLinear(ResidualAdapter):
    def __init__(self, base_module:nn.Linear):
        super(HeadLinear, self).__init__(base_module)

    def forward_adapter(self, x, **kwargs):
        return self.base_module(x, **kwargs)

    def forward_base(self, x, **kwargs):
        return 0

    def forward_merge(self, x, **kwargs):
        return self.base_module(x, **kwargs)


def inject_residual_adapter(model, adapter='LoRAmul_VPTadd', rank=10):
    AdapterClass = LoRAmul_VPTadd_Linear
    if adapter == 'LoRAadd': AdapterClass = LoRAadd_Linear
    elif adapter == 'VPTadd': AdapterClass = VPTadd_Linear

    for name, l in model.named_modules():
        if isinstance(l, nn.Linear):
            parent_layer = model
            tokens = name.strip().split('.')
            for t in tokens[:-1]:
                parent_layer = parent_layer[int(t)] if t.isnumeric() else getattr(parent_layer, t)
            linear = getattr(parent_layer, tokens[-1])
            linear_adapter = HeadLinear(linear) if 'head' in tokens else AdapterClass(linear, rank=rank)
            setattr(parent_layer, tokens[-1], linear_adapter)

def get_adapters_and_block_ids(model):
    """
    Returns a list of [parent_layer, adapter_name, block_id] for each ResidualAdapter.
    The block_id of the classification head is set to max block_id + 1.
    """
    adapters_and_block_ids = []
    for name, l in model.named_modules():
        if isinstance(l, ResidualAdapter):
            parent_layer = model
            tokens = name.strip().split('.')
            block_key = []
            for t in tokens[:-1]:
                parent_layer = parent_layer[int(t)] if t.isnumeric() else getattr(parent_layer, t)
                if t.isnumeric(): block_key.append(int(t))
            if 'head' in name: block_key = [float('inf')]
            adapters_and_block_ids.append([parent_layer, tokens[-1], block_key])

    max_len= max(len(bk) for _, _, bk in adapters_and_block_ids)
    adapters_and_block_ids = [[pl, tk, tuple(bk+[0] * (max_len-len(bk)))] for pl, tk, bk in adapters_and_block_ids]
    block_id_map = {block_key: i for i, block_key in enumerate(sorted(set(bk for _, _, bk in adapters_and_block_ids)))}

    for i, (_, _, block_key) in enumerate(adapters_and_block_ids):
        adapters_and_block_ids[i][2] = block_id_map[block_key]

    return adapters_and_block_ids
