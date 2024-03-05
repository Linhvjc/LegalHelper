from __future__ import annotations

import torch


def generate_bow(input_ids, output_dim, device, values=None):
    """from a batch of input ids, generates batch of bow rep"""
    bs = input_ids.shape[0]
    bow = torch.zeros(bs, output_dim).to(device)
    if values is None:
        bow[torch.arange(bs).unsqueeze(-1), input_ids] = 1
    else:
        bow[torch.arange(bs).unsqueeze(-1), input_ids] = values
    return bow


def normalize(tensor, eps=1e-9):
    """normalize input tensor on last dimension"""
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)
