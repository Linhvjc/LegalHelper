from __future__ import annotations

import torch.nn as nn


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding based on the specified pooling type.
    """

    VALID_POOLER_TYPES = {
        'cls', 'cls_before_pooler',
        'avg', 'avg_top2', 'avg_first_last',
    }

    def __init__(self, pooler_type):
        super().__init__()
        assert pooler_type in self.VALID_POOLER_TYPES, f'unrecognized pooling type {pooler_type}'
        self.pooler_type = pooler_type

    def forward(self, attention_mask, outputs):
        if self.pooler_type == 'cls_before_pooler' or self.pooler_type == 'cls':
            return self._cls_pooling(outputs.last_hidden_state)
        elif self.pooler_type == 'avg':
            return self._avg_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooler_type == 'avg_first_last':
            return self._avg_first_last_pooling(outputs.hidden_states, attention_mask)
        elif self.pooler_type == 'avg_top2':
            return self._avg_top2_pooling(outputs.hidden_states, attention_mask)
        else:
            raise NotImplementedError

    @staticmethod
    def _cls_pooling(last_hidden):
        return last_hidden[:, 0]

    @staticmethod
    def _avg_pooling(last_hidden, attention_mask):
        return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            -1,
        ).unsqueeze(-1)

    @staticmethod
    def _avg_first_last_pooling(hidden_states, attention_mask):
        first_hidden = hidden_states[1]
        last_hidden = hidden_states[-1]
        return ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
            1,
        ) / attention_mask.sum(-1).unsqueeze(-1)

    @staticmethod
    def _avg_top2_pooling(hidden_states, attention_mask):
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        return ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
            1,
        ) / attention_mask.sum(-1).unsqueeze(-1)
