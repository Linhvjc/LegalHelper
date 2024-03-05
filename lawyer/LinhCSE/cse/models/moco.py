from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from transformers import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from .modules import MLPLayer
from .modules import Pooler
from cse.losses.mid_level import Divergence
from cse.models.modules.similarity import SimilarityFunctionWithTorch


class SiameseModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [
        r'lm_head.decoder.weight', r'lm_head.decoder.bias',
    ]
    _keys_to_ignore_on_load_missing = [
        r'position_ids',
        r'lm_head.decoder.weight',
        r'lm_head.decoder.bias',
    ]

    def __init__(self, config: PretrainedConfig, args):
        super().__init__(config)

        self.args = args
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.lm_head = RobertaLMHead(config)

        self.pooler = Pooler(self.args.pooler_type)
        self.mlp = MLPLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        **kwargs,
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        mlp_outputs = self.mlp(outputs.last_hidden_state)
        outputs.last_hidden_state = mlp_outputs
        pooled_output = self.pooler(attention_mask, outputs)

        return outputs, pooled_output

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
    ):
        _, pooled_output = self.get_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return pooled_output


class Moco(nn.Module):
    def __init__(
        self,
        config,
        args,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Initializes an instance of the MocoRankcse class.

        Args:
            config: Configuration object.
            args: Arguments object.
            label_smoothing: The number of label smoothing from 0 to 1, Default 0.0
        """

        super().__init__()

        self.args = args
        self.retriever = SiameseModel.from_pretrained(
            self.args.model_name_or_path, args=args,
        )
        self.encoder_q = self.retriever
        self.encoder_k = copy.deepcopy(self.retriever)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer(
            'queue', torch.randn(
                self.args.projection_size, self.args.queue_size,
            ),
        )
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # RankCSE parameter
        self.label_smoothing = label_smoothing
        self.mlp = MLPLayer(config)
        self.sim = SimilarityFunctionWithTorch()
        self.div = Divergence(beta_=self.args.beta_)
        self.pooler = Pooler(self.args.pooler_type)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _momentum_update_key_encoder(self):
        """
        Update the key encoder by applying momentum to the parameters.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.args.momentum + param_q.data * (
                1.0 - self.args.momentum
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Dequeue and enqueue keys into the queue buffer.

        Args:
            keys: Keys to be enqueued.
        """

        batch_size = keys.shape[0]
        assert (
            self.args.queue_size % batch_size == 0
        ), f'{batch_size}, {self.args.queue_size}'  # for simplicity
        self.queue[
            :, :self.args.queue_size -
            batch_size,
        ] = self.queue[:, batch_size:]
        self.queue[:, self.args.queue_size - batch_size:] = keys.T

    def _compute_logits(self, q, k):
        """
        Compute logits for the contrastive loss.

        Args:
            q: Queries.
            k: Keys.

        Returns:
            Computed logits.
        """

        l_pos = cosine_similarity(q, k).unsqueeze(-1)
        batch_neg = self.queue.clone().detach().transpose(0, 1)
        l_neg = self.sim(batch_neg, q)

        logits = torch.cat([l_pos, l_neg], dim=1)
        return logits

    def forward(self, input_ids, attention_mask, teacher_pred):
        """
        Forward pass of the MocoRankcse model.

        Args:
            input_ids: Input IDs.
            attention_mask: Attention mask.
            teacher_pred: Teacher predictions.

        Returns:
            MocoOutput object containing the computed loss.
        """

        # !Contrastive loss
        split_ids = torch.split(input_ids, split_size_or_sections=1, dim=1)
        split_mask = torch.split(
            attention_mask, split_size_or_sections=1, dim=1,
        )

        q_tokens, k_tokens = split_ids[0], split_ids[1]
        q_mask, k_mask = split_mask[0], split_mask[1]

        # Flatten input for encoding
        q_tokens = q_tokens.view((-1, q_tokens.shape[-1]))
        k_tokens = k_tokens.view((-1, k_tokens.shape[-1]))
        q_mask = q_mask.view((-1, q_mask.shape[-1]))
        k_mask = k_mask.view((-1, k_mask.shape[-1]))

        input_ids = input_ids.view((-1, input_ids.shape[-1]))
        attention_mask = attention_mask.view((-1, attention_mask.shape[-1]))

        pooled_q = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask)
        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()

            if not self.encoder_k.training:
                self.encoder_k.eval()

            pooled_k = self.encoder_k(
                input_ids=k_tokens, attention_mask=k_mask,
            )
        logits = self._compute_logits(pooled_q, pooled_k) / self.args.temp

        # labels: positive key indicators
        labels = torch.zeros(
            logits.shape[0], dtype=torch.long,
        ).to(q_tokens.device)

        loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            label_smoothing=0.05,
        )
        self._dequeue_and_enqueue(pooled_k)

        return loss

    def save_pretrained(self, output_dir: str | None = None):
        """
        Save the pretrained model to the specified output directory.

        Args:
            output_dir: Output directory for saving the model.
        """

        self.retriever.save_pretrained(output_dir)
