import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

from utils import logger


class NLIFinetuneModel(nn.Module):
    # Data structure: sent_1, sent_2, hard_neg
    def __init__(self, model_path_or_name: str, temp: float, queue_len: int, pooler_type: str):
        super(NLIFinetuneModel, self).__init__()

        self.student_model = AutoModel.from_pretrained(model_path_or_name)
        self.logit_scale = nn.Parameter(torch.tensor([temp], dtype=torch.float64))

        self.temp = temp
        self.queue_len = queue_len
        self.pooler_type = pooler_type

    def forward(self, queue=None, **sample_zh):
        if self.pooler_type == 'cls':
            zh_features = self.student_model(**sample_zh, output_hidden_states=True, return_dict=True).pooler_output
        elif self.pooler_type == 'cbp':
            zh_features = self.student_model(**sample_zh, output_hidden_states=True,
                                             return_dict=True).last_hidden_state[:, 0]
        zh_features = zh_features.view(-1, 3, zh_features.shape[-1])
        key, z2, z3 = zh_features[:, 0], zh_features[:, 1], zh_features[:, 2]
        query = torch.cat([torch.cat([z2, z3], dim=0), queue], dim=0)
        labels = torch.arange(key.shape[0])
        key = key / key.norm(dim=-1, keepdim=True)
        query = query / query.norm(dim=-1, keepdim=True)

        if self.temp <= 5.01:
            scores = self.logit_scale.exp() * torch.einsum('ab,cb->ac', key, query)
        else:
            scores = self.logit_scale * torch.einsum('ab,cb->ac', key, query)
        loss = F.cross_entropy(scores, labels.to(scores.device))
        queue = query[:query.shape[0] - max(query.shape[0] - self.queue_len, 0)]

        return loss, queue.detach().cpu(), self.logit_scale


class NLI_CLS_Model(nn.Module):
    # Data structure: sentence1, sentence2, annotator_label, gold_label
    def __init__(self,
                 model_path_or_name: str,
                 num_labels: int,
                 pooler_type: str = "cls",
                 dropout_prob: float = 0.1):
        super(NLI_CLS_Model, self).__init__()

        self.bert = AutoModel.from_pretrained(model_path_or_name)
        self.dropout = nn.Dropout(dropout_prob)
        out_features = 1024
        try:
            out_features = self.bert.pooler.dense.out_features
        except:
            logger.error(f"Not get out_features of teacher model")

        self.classifier = nn.Linear(out_features, num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.pooler_type = pooler_type

    def forward(self, **inputs):
        features = self.bert(**inputs, output_hidden_states=True, return_dict=True)
        if self.pooler_type == 'cls':
            features = features.pooler_output
        elif self.pooler_type == 'cbp':
            features = features.last_hidden_state[:, 0]

        output = self.dropout(features)
        logits = self.classifier(output)
        probs = self.softmax(logits)

        return logits, probs

    def encode(self, **inputs):
        features = self.bert(**inputs, output_hidden_states=True, return_dict=True)
        if self.pooler_type == 'cls':
            features = features.pooler_output
        elif self.pooler_type == 'cbp':
            features = features.last_hidden_state[:, 0]
        return features
