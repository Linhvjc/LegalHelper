from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

class MLPLayer(nn.Module):
    """
    Head for obtaining sentence representations over transformers' representation.
    Configures an MLP with a specific architecture, including residual connections.
    """

    def __init__(self, config):
        super().__init__()
        # Use a list to store layers if you have a fixed number of layers
        self.layers = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(3)
            ],
        )

        # Use a dictionary to store activation functions if you have a fixed pattern
        self.activations = {0: nn.GELU(), 1: nn.GELU(), 2: nn.ReLU()}

    def forward(self, features, **kwargs):
        x = features
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x))
        return (
            features + x
        )  # Assuming you want to add the input features as a residual connection to the output

model_name = '/home/link/DistilCSE/eval/base'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
config = AutoConfig.from_pretrained(model_name)
mlp_layer = MLPLayer(config)
model.classifier = mlp_layer

# for param in model.parameters():
#     param.requires_grad = False

# class PhoBERTWithMLP(nn.Module):
#     def __init__(self, phobert_model, mlp_layer):
#         super(PhoBERTWithMLP, self).__init__()
#         self.phobert = phobert_model
#         self.mlp = mlp_layer

#     def forward(self, input_ids, attention_mask):
#         outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         logits = self.mlp(pooled_output)
#         return logits

# model_with_mlp = PhoBERTWithMLP(model, mlp_layer)
model.save_pretrained('/home/link/DistilCSE/eval/out')
