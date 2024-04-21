from torch import nn
from transformers import AutoModel


class LinearBert(nn.Module):
    def __init__(self, model_name, config, pooler_type=None, out_features: int = 1024):
        super(LinearBert, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name, config=config)
        if config.num_hidden_layers == 6 or config.num_hidden_layers == 12:
            self.linear = nn.Linear(768, out_features)
        elif config.num_hidden_layers == 4:
            self.linear = nn.Linear(312, out_features)

        self.pooler_type = pooler_type

    def forward(self, **x):
        if 'cls' in self.pooler_type:
            x = self.linear(self.bert(**x).pooler_output)
        elif 'cbp' in self.pooler_type:
            x = self.linear(self.bert(**x).last_hidden_state[:, 0])
        return x
