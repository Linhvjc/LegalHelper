import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import AutoConfig


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class SwiGLURoberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.hidden_dropout_prob)
        self.intermediate_act_fn = SwiGLU()
        self.intermediate_dense = nn.Linear(
            config.intermediate_size//2, config.intermediate_size, bias=config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dense(hidden_states)
        return hidden_states
