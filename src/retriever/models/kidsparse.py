from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.models.roberta.modeling_roberta import RobertaModel

from .modules import SimilarityFunctionWithTorch
from src.retriever.schemas.outputs import KidSparseOutput


class ConLearnSparseRoberta(RobertaForMaskedLM):
    def __init__(self, config: PretrainedConfig, args):
        super().__init__(config)

        self.args = args

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ['lm_head.decoder.weight'])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_output(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        is_query: bool = False,
        **kwargs,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        # (bs, max_seq, vocab_size)
        decoded_output = self.lm_head(sequence_output)

        repr_input = None

        if is_query:
            pass  # TODO
        else:
            if self.args.passage_wise_type == 'sum':
                repr_input = torch.sum(
                    torch.log(1 + torch.relu(decoded_output)) * attention_mask.unsqueeze(-1), dim=1,
                )
            else:  # max
                repr_input, _ = torch.max(
                    torch.log(1 + torch.relu(decoded_output)) * attention_mask.unsqueeze(-1), dim=1,
                )
                # 0 masking also works with max because all activations are positive

        return decoded_output, repr_input

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        input_ids_positive: torch.LongTensor | None = None,
        attention_mask_positive: torch.FloatTensor | None = None,
        input_ids_negative: torch.LongTensor | None = None,
        attention_mask_negative: torch.FloatTensor | None = None,
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
        # TODO: use contrastive loss for sparse embedding. The original SPLADE paper used diffrent from loss
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
        return KidSparseOutput(
            total_loss=total_loss,
            loss_ct=loss_ct,
            loss_ct_dpi_query=loss_ct_dpi_query,
            loss_ct_dpi_positive=loss_ct_dpi_positive,
        )
