"""
Implementation of adding noise and applying consistency regularization.
"""
import torch
import torch.nn as nn
from typing import List
from .residual_adapters import ResidualAdapter
from .sharable_dropout import set_duplicate

class MultiplicativeNoiseAdapter(nn.Module):
    def __init__(self, adapter:ResidualAdapter, sigma=1.,
                 shape:str='BC', # 'BC' or 'BTC',
                 ):
        super(MultiplicativeNoiseAdapter, self).__init__()
        self.adapter = adapter
        self.sigma = sigma
        self.shape = shape

    def _get_noise_shape(self, x:torch.Tensor) -> List[int] or torch.Size:
        if len(x.shape) >= 2:
            if self.shape == 'BC':
                return [x.shape[0]] + [1] * (x.ndim - 2) + [x.shape[-1]]
            else:
                return x.shape
        else:
            raise NotImplementedError

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.training and self.sigma > 0:
            base_feature = self.adapter.forward_base(x)
            adapter_feature = self.adapter.forward_adapter(x)
            with torch.no_grad():
                noise_shape = self._get_noise_shape(adapter_feature)
                noise = torch.randn(*noise_shape, device=adapter_feature.device) * self.sigma + 1
            return base_feature + adapter_feature * noise
        else:
            return self.adapter.forward(x)

    def extra_repr(self) -> str:
        return f'sigma={self.sigma}, shape={self.shape}'


class PACE_MSELoss(nn.Module):
    def forward(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return (input-target).square().sum(dim=-1).mean()

_mse_criterion = PACE_MSELoss()

def compute_loss_pace(model, x, y, criterion, lbd_pace=1, pace_criterion=_mse_criterion, **kwargs):
    # duplicate the input, e.g [x1, x2, x3] will be [x1, x2, x3, x1, x2, x3]
    set_duplicate(model, 2)
    x_duplicate = torch.cat([x, x])
    logits = model(x_duplicate)
    logits_1, logits_2 = torch.chunk(logits, chunks=2)
    cls_loss = criterion(logits_1, y)
    pace_loss = pace_criterion(logits_1, logits_2)
    total_loss = cls_loss + lbd_pace * pace_loss
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss, 'total_loss': total_loss, 'logits': logits_1}

def compute_loss_pace_lazy_half(model, x, y, criterion, lbd_pace=1, pace_criterion=_mse_criterion, itr=0, lazy_interval=2, **kwargs):
    bs = x.shape[0]
    results_dict = {}
    if itr % lazy_interval == 0:
        bs_half = bs // 2
        set_duplicate(model, 2)
        x_duplicate = torch.cat([x[:bs_half], x[:bs_half]])
        logits = model(x_duplicate)
        logits_1, logits_2 = torch.chunk(logits, chunks=2)
        cls_loss = criterion(logits_1, y[:bs_half])
        pace_loss = pace_criterion(logits_1, logits_2)
        total_loss = cls_loss + lbd_pace * pace_loss
        results_dict['pace_loss'] = pace_loss
    else:
        set_duplicate(model, 1)
        logits_1 = model(x)
        cls_loss = criterion(logits_1, y)
        total_loss = cls_loss
    results_dict.update({'cls_loss': cls_loss, 'total_loss': total_loss, 'logits': logits_1})
    return results_dict


def compute_loss_pace_fast(model, x, y, criterion, lbd_pace=1, pace_criterion=_mse_criterion, history_logits=None, index=None, **kwargs):
    set_duplicate(model, 1)
    logits = model(x)
    cls_loss = criterion(logits, y)
    with torch.no_grad():
        logits_recent = history_logits[index].to(logits.device)
        history_logits[index] = logits.to(history_logits.device)
    pace_loss = pace_criterion(logits, logits_recent)
    total_loss = cls_loss + lbd_pace * pace_loss
    return {'cls_loss': cls_loss, 'pace_loss': pace_loss, 'total_loss': total_loss, 'logits': logits}

