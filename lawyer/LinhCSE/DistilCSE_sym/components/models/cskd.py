import torch
from torch import nn
import torch.nn.functional as F

from utils import logger
# from modules.mlp import MLPLayer

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


class ContrastiveKD(nn.Module):
    def __init__(self,
                 student_model=None,
                 teacher_model=None,
                 args=None,
                 freeze=True):
        super(ContrastiveKD, self).__init__()

        self.student_model = student_model
        self.teacher_model = teacher_model
        # self.mlp = MLPLayer()

        if freeze == True:
            out_features = 1024
            try:
                out_features = self.teacher_model.pooler.dense.out_features
            except:
                logger.error(f"Not get out_features of teacher model")

            for params in self.teacher_model.named_parameters():
                params[1].require_grad = False

            num_hidden_layers = self.student_model.config.num_hidden_layers
            if num_hidden_layers != 4:
                self.linear = nn.Linear(768, out_features)
            else:
                self.linear = nn.Linear(312, out_features)
        else:
            self.linear = nn.Linear(768, 768)
        self.logit_scale = nn.Parameter(torch.tensor([args.temp]))
        self.temp_exp = args.temp_exp
        self.pooler_type = args.pooler_type
        self.linear_2 = nn.Linear(768, 1024)
        self.mlp = MLPLayer(student_model.config)

    def forward(self, 
                student_inputs_query, 
                student_inputs_document,
                teacher_inputs_query,
                teacher_inputs_document,
                queue_query, 
                queue_document, 
                steps, queue_len, mse):
        if self.pooler_type == 'cls':
            # student_features = self.student_model(**student_inputs, output_hidden_states=True,
            #                                       return_dict=True).pooler_output
            student_features_query = self.student_model(**student_inputs_query, output_hidden_states=True,
                                                  return_dict=True).pooler_output
            student_features_document = self.student_model(**student_inputs_document, output_hidden_states=True,
                                                  return_dict=True).pooler_output

        # student_features = self.mlp(student_features)
        student_features_query = self.mlp(student_features_query)
        student_features_document = self.mlp(student_features_document)

        student_features_query = self.linear_2(student_features_query)
        student_features_document = self.linear_2(student_features_document)
        with torch.no_grad():
            if self.pooler_type == 'cls':
                # teacher_features = self.teacher_model(**teacher_inputs, output_hidden_states=True,
                #                                       return_dict=True).pooler_output
                teacher_features_query = self.teacher_model(**teacher_inputs_query, output_hidden_states=True,
                                                      return_dict=True).pooler_output
                teacher_features_document = self.teacher_model(**teacher_inputs_document, output_hidden_states=True,
                                                      return_dict=True).pooler_output
        # teacher_features_query = self.linear_2(teacher_features_query)
        # teacher_features_document = self.linear_2(teacher_features_document)

        if self.temp_exp == 1:
            temp = self.logit_scale.exp()
        else:
            temp = self.logit_scale

        # loss, teacher_queue = self.criterion(student_features, teacher_features, temp, queue, steps,
        #                                      queue_len=queue_len)
        loss_query, teacher_queue_query = self.criterion(student_features_query, 
                                                         teacher_features_query, 
                                                         temp, queue_query, steps,
                                                            queue_len=queue_len)

        loss_document, teacher_queue_document = self.criterion(student_features_document, 
                                                               teacher_features_document, 
                                                               temp, queue_document, steps,
                                                                queue_len=queue_len)
        loss = loss_query + 0.5* loss_document
        if mse == 1:
            loss_mse_query = nn.MSELoss()(student_features_query, teacher_features_query)
            loss_mse_document = nn.MSELoss()(student_features_document, teacher_features_document)
            loss_mse = loss_mse_query + loss_mse_document
            loss += loss_mse
        return loss, teacher_queue_query, teacher_queue_document, temp

    def criterion(self, query, key, temp, queue, steps, queue_len=20000):
        labels = torch.arange(key.shape[0])
        key = key / key.norm(dim=-1, keepdim=True)
        query = query / query.norm(dim=-1, keepdim=True)
        key = torch.cat([key, queue.to(query.device)], dim=0)
        scores = temp * torch.einsum('ab,cb->ac', query, key)
        loss = F.cross_entropy(scores, labels.to(scores.device))
        queue = key[:key.shape[0] - max(key.shape[0] - queue_len, 0)]
        return loss, queue.detach().cpu()
