from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from .modules import Pooler
from .modules import SimilarityFunctionWithTorch
from retriever.schemas.outputs import KidDenseOutput


class KidDenseRoberta(RobertaPreTrainedModel):
    def __init__(self, config: PretrainedConfig, args):
        super().__init__(config)

        self.args = args
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.lm_head = RobertaLMHead(config)

        self.pooler = Pooler(self.args.pooler_type)
        # self.mlp = MLPLayer(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        # self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    # def get_output_embeddings(self):
    #     return self.lm_head.decoder

    # def set_output_embeddings(self, new_embeddings):
    #     self.lm_head.decoder = new_embeddings

    def get_output(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        **kwargs,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = self.pooler(attention_mask, outputs)

        return outputs, pooled_output

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        input_ids_positive: torch.LongTensor | None = None,
        attention_mask_positive: torch.FloatTensor | None = None,
        input_ids_negative: torch.LongTensor | None = None,
        attention_mask_negative: torch.FloatTensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        _, pooled_output = self.get_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if not kwargs.get('is_train', ''):
            return pooled_output

        # init loss
        total_loss = 0.0
        loss_ct = 0.0
        loss_ct_dpi_query = 0.0
        loss_ct_dpi_positive = 0.0

        _, pooled_output_positive = self.get_output(
            input_ids=input_ids_positive,
            attention_mask=attention_mask_positive,
        )

        sim_fn = SimilarityFunctionWithTorch(self.args.sim_fn)
        scores = sim_fn(pooled_output, pooled_output_positive)

        if input_ids_negative and attention_mask_negative:
            _, pooled_output_negative = self.get_output(
                input_ids=input_ids_negative,
                attention_mask=attention_mask_negative,
            )
            scores_negative = sim_fn(pooled_output, pooled_output_negative)
            # hard negative in batch negative
            scores = torch.cat([scores, scores_negative], 1)

            weights = torch.tensor(
                [
                    [0.0] * (scores.size(-1) - scores_negative.size(-1))
                    + [0.0] * i
                    + [self.args.weight_hard_negative]
                    + [0.0] * (scores_negative.size(-1) - i - 1)
                    for i in range(scores_negative.size(-1))
                ],
            ).to(pooled_output.device)
            scores = scores + weights

        # Contrastice Learning Stratege - InBatch
        labels = torch.arange(scores.size(0)).long().to(pooled_output.device)
        loss_fct = nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing,
        )

        loss_ct = loss_fct(scores, labels)
        total_loss += loss_ct

        if self.args.dpi_query:
            _, pooled_output_dropout = self.get_output(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            scores = sim_fn(pooled_output, pooled_output_dropout)

            labels = torch.arange(scores.size(0)).long().to(
                pooled_output.device,
            )
            loss_ct_dpi_query = loss_fct(scores, labels)
            total_loss += self.args.coff_dpi_query * loss_ct_dpi_query

        if self.args.dpi_positive:
            _, pooled_output_positive_dropout = self.get_output(
                input_ids=input_ids_positive,
                attention_mask=attention_mask_positive,
            )
            scores = sim_fn(
                pooled_output_positive,
                pooled_output_positive_dropout,
            )

            labels = torch.arange(scores.size(0)).long().to(
                pooled_output_positive.device,
            )
            loss_ct_dpi_positive = loss_fct(scores, labels)
            total_loss += self.args.coff_dpi_positive * loss_ct_dpi_positive

        loss_ct_dpi = 0.5 * loss_ct_dpi_query + 0.5 * loss_ct_dpi_positive

        if not return_dict:
            return (total_loss, loss_ct, loss_ct_dpi, scores)

        return KidDenseOutput(
            total_loss=total_loss,
            loss_ct=loss_ct,
            loss_ct_dpi=loss_ct_dpi,
            logits=scores,
        )
