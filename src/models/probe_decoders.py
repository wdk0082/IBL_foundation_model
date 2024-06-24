import torch
from torch import nn
import numpy as np

class ProbeDecoder(nn.Module):
    def __init__(
            self,
            input_feature_size,
            input_seq_len,
            output_size,
            config,
    ):
        super().__init__()
        self.config = config

        # For the token dimension
        input_dim = input_feature_size
        if config.merge_type == 'concat':
            input_dim *= input_seq_len
        elif config.merge_type == 'weighted_avg':
            self.weights = nn.Parameter(torch.randn(input_seq_len))
        elif config.merge_type == 'avg':
            pass

        # main decoder
        if config.type == 'mlp':
            ACTIVATION = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'sigmoid': nn.Sigmoid,
                'gelu': nn.GELU,
            }
            act_fn = ACTIVATION[config.act]
            decoder_list = [
                nn.Linear(input_dim, config.hidden_size),
                act_fn(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, output_size),
            ]
            self.decoder = nn.Sequential(*decoder_list)
        elif config.type == 'linear':
            self.decoder = nn.Linear(input_dim, output_size)

    def forward(self, x):  # (bs, seq_len, feature_size)

        if self.config.merge_type == 'concat':
            x = x.view(x.size(0), -1)
        elif self.config.merge_type == 'weighted_avg':
            x = torch.sum(x * self.weights.unsqueeze(0).unsqueeze(-1), dim=1)
        elif self.config.merge_type == 'avg':
            x = torch.mean(x, dim=1)

        return self.decoder(x)
