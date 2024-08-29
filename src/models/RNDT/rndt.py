import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from iblatlas.regions import BrainRegions

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import MLP

from transformers.activations import ACT2FN

ACT2FN['softsign'] = nn.Softsign

from src.utils.config_utils import DictConfig, update_config
from src.models.model_output import ModelOutput
from src.models.masker import Masker

DEFAULT_CONFIG = "src/configs/rndt/rndt.yaml"


@dataclass
class RNDTOutput(ModelOutput):
    loss: Optional[torch.LongTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    preds: Optional[torch.LongTensor] = None
    targets: Optional[torch.LongTensor] = None


# Create buffer of biggest possible context mask (dtype=torch.float32)
def create_context_mask(context_forward, context_backward, max_F) -> Tensor:  # (max_seq_len, max_seq_len)

    # Initialize mask with zeros
    mask = torch.zeros(max_F, max_F).to(torch.float32)

    # If both context_forward and context_backward are -1, set entire mask to zero (i.e., no masking)
    if context_forward == -1 and context_backward == -1:
        return mask

    # Define context range
    context_forward = context_forward if context_forward >= 0 else max_F
    context_backward = context_backward if context_backward >= 0 else max_F

    # Create forward context mask
    forward_mask = torch.triu(torch.ones(max_F, max_F), diagonal=-context_forward).to(torch.float32).transpose(0, 1)

    # Apply backward context mask if necessary
    if context_backward > 0:
        backward_mask = torch.triu(torch.ones(max_F, max_F), diagonal=-context_backward).to(torch.float32)
        forward_mask = forward_mask * backward_mask

    # Set masked positions to -inf
    mask[forward_mask == 0] = float('-inf')

    return mask


# Create buffer of biggest possible sinusoidal positional embedding
def create_sinusoidal_pe(max_F, hidden_size) -> torch.LongTensor:  # (max_F, hidden_size)
    position_encoding = torch.zeros(max_F, hidden_size)
    position = torch.arange(0, max_F, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size))
    position_encoding[:, 0::2] = torch.sin(position * div_term)
    position_encoding[:, 1::2] = torch.cos(position * div_term)
    return position_encoding


class RegionManager:
    def __init__(self, str_truncate=0):
        full_brain_regions = BrainRegions().acronym

        # use str_truncate = 0 to skip truncation
        if str_truncate > 0:
            truncate = np.vectorize(lambda x: x[:str_truncate])
            truncated_regions = truncate(full_brain_regions)
        else:
            truncated_regions = full_brain_regions

        self.brain_regions = np.unique(truncated_regions)
        self.region_to_index = {region: index for index, region in enumerate(self.brain_regions)}
        self.index_to_region = {index: region for index, region in enumerate(self.brain_regions)}
        # special case for nan (padding)
        self.region_to_index['nan'] = len(self.brain_regions)
        self.index_to_region[len(self.brain_regions)] = 'nan'

    def __len__(self):
        return len(self.brain_regions)


class NeuralMLP(nn.Module):

    def __init__(self, hidden_size, inter_size, act, use_bias, dropout):
        super().__init__()

        self.up_proj = nn.Linear(hidden_size, inter_size, bias=use_bias)
        self.act = ACT2FN[act]
        self.down_proj = nn.Linear(inter_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.up_proj(x))
        return self.dropout(self.down_proj(x))


class ScaleNorm(nn.Module):

    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class NeuralEmbedder(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            config: DictConfig,
    ):
        super().__init__()
        self.config = config
        if config.embedder_type == 'linear':
            self.embedder = nn.Linear(config.n_channel, hidden_size)
        elif config.embedder_type == 'identity':
            self.embedder = nn.Identity()
        else:
            raise NotImplementedError(f"Embedder type {config.embedder_type} not implemented")

        self.activation = ACT2FN[config.activation] if config.activation != 'identity' else nn.Identity()

        # PE
        self.pos = config.pos
        if self.pos == 'learnable':
            self.embed_pos = nn.Embedding(config.max_F, hidden_size)
        elif self.pos is None:
            pass
        elif self.pos == 'sinusoidal':
            sin_pos = create_sinusoidal_pe(config.max_F, hidden_size)
            self.register_buffer('sin_pos', sin_pos, persistent=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self,
            spikes: torch.FloatTensor,  # (bs, seq_len, n_channel)
            spikes_timestamps: Optional[torch.FloatTensor] = None,  # (bs, seq_len)
    ) -> torch.FloatTensor:

        # Embedder
        x = self.activation(self.embedder(spikes))

        # TODO: Do we need a different Normalization for each session here? Or do we need Normalization at all?

        # PE
        if self.pos == 'learnable':
            x += self.embed_pos(spikes_timestamps)
        elif self.pos == 'sinusoidal':
            cur_pe = self.sin_pos[:x.size(1), :].to(x.device).unsqueeze(0).expand(x.size(0), x.size(1), x.size(2))
            x += cur_pe
        elif self.pos is None:
            pass

        return self.dropout(x)


class NeuralEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = None,
            dropout: float = 0.1,
            bias: bool = True,
            activation: str = "gelu",
            norm: str = "layernorm",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = NeuralMLP(d_model, dim_feedforward, activation, bias, dropout)
        self.norm1 = ScaleNorm(d_model ** 0.5) if norm == "scalenorm" else nn.LayerNorm(d_model)
        self.norm2 = ScaleNorm(d_model ** 0.5) if norm == "scalenorm" else nn.LayerNorm(d_model)

    def forward(
            self,
            src: torch.FloatTensor,
            src_mask: torch.FloatTensor = None,
            padding_mask: torch.BoolTensor = None,
            output_attentions: bool = False,
    ):
        src2, attn_weights = self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src), attn_mask=src_mask, key_padding_mask=padding_mask,
                                            average_weights=False)
        src = src + src2
        src = src + self.ffn(self.norm2(src))
        if output_attentions:
            return src, attn_weights
        else:
            return src


class NeuralDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = None,
            dropout: float = 0.1,
            bias: bool = True,
            activation: str = "gelu",
            norm: str = "layernorm",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = NeuralMLP(d_model, dim_feedforward, activation, bias, dropout)
        self.norm1 = ScaleNorm(d_model ** 0.5) if norm == "scalenorm" else nn.LayerNorm(d_model)
        self.norm2 = ScaleNorm(d_model ** 0.5) if norm == "scalenorm" else nn.LayerNorm(d_model)
        self.norm3 = ScaleNorm(d_model ** 0.5) if norm == "scalenorm" else nn.LayerNorm(d_model)

    def forward(
            self,
            src: torch.FloatTensor,
            tgt: torch.FloatTensor,
            src_mask: torch.FloatTensor = None,
            tgt_mask: torch.FloatTensor = None,
            src_padding_mask: torch.BoolTensor = None,
            tgt_padding_mask: torch.BoolTensor = None,
            output_attentions: bool = False,
    ):
        tgt2, self_attn_weights = self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt), attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask,
                                            average_weights=False)
        tgt = tgt + tgt2
        tgt2, cross_attn_weights = self.cross_attn(self.norm2(tgt), self.norm2(src), self.norm2(src), attn_mask=src_mask, key_padding_mask=src_padding_mask,
                                            average_weights=False)
        tgt = tgt + tgt2
        tgt = tgt + self.ffn(self.norm3(tgt))
        if output_attentions:
            return tgt, self_attn_weights, cross_attn_weights
        else:
            return tgt



class NeuralFactorsProjection(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()

        self.out_size = config.size if config.active else hidden_size
        self.dropout = nn.Dropout(config.dropout)
        if config.active:
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, config.size, config.bias),
                ACT2FN[config.act]
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(self.dropout(x))


class NeuralEncoder(nn.Module):
    def __init__(
            self,
            config: DictConfig,
    ):
        super().__init__()
        self.config = config
        self.spikes_embedder = NeuralEmbedder(config.transformer.hidden_size, config.embedder)
        self.transformer_blocks = nn.ModuleList(
            [
                NeuralEncoderLayer(
                    d_model=config.transformer.hidden_size,
                    nhead=config.transformer.n_head,
                    dim_feedforward=config.transformer.dim_feedforward,
                    dropout=config.transformer.dropout,
                    bias=config.transformer.bias,
                    activation=config.transformer.activation,
                    norm=config.transformer.norm,
                )
                for _ in range(config.transformer.n_layers)
            ]
        )
        self.out_norm = ScaleNorm(
            config.transformer.hidden_size ** 0.5) if config.transformer.norm == "scalenorm" else nn.LayerNorm(
            config.transformer.hidden_size)
        self.factors_projection = NeuralFactorsProjection(config.transformer.hidden_size, config.factors_projection)

        # attention mask
        context_mask = create_context_mask(config.context_forward, config.context_backward, config.embedder.max_F)
        self.register_buffer('context_mask', context_mask, persistent=False)

    def forward(
            self,
            spikes: torch.FloatTensor,  # (bs, seq_len, n_channel)
            spikes_timestamps: Optional[torch.FloatTensor] = None,  # (bs, seq_len)
            padding_mask: Optional[torch.LongTensor] = None,  # (bs, seq_len)
    ) -> torch.FloatTensor:
        if self.config.encoder_type == 'NDT1':
            x = spikes
        elif self.config.encoder_type == 'iTransformer':
            x = spikes.transpose(1, 2)
        else:
            raise NotImplementedError(f"Encoder type {self.config.encoder_type} not implemented")

        # Embedder
        x = self.spikes_embedder(x, spikes_timestamps)

        # Transformer
        # TODO: It's assumed that all samples in one batch has the same length. Change context mask to (bs x n_heads, seq_len, seq_len) to generalize.
        cur_context_mask = self.context_mask[:x.size(1), :x.size(1)].to(x.device)  # (seq_len, seq_len), float32
        padding_mask = ~padding_mask.to(torch.bool) if padding_mask is not None else torch.zeros(x.size(0), x.size(1)).to(torch.bool)
        for layer in self.transformer_blocks:
            x = layer(
                x,
                src_mask=cur_context_mask,  # (seq_len, seq_len), float32
                padding_mask=padding_mask,  # (bs, seq_len), bool
            )
        x = self.out_norm(x)

        # Factors projection
        x = self.factors_projection(x)

        return x


class NeuralRegionQuery(nn.Module):
    def __init__(
            self,
            config: DictConfig,
            region_to_idx: Dict,
    ):
        super().__init__()
        self.config = config
        self.region_to_idx = region_to_idx
        self.n_regions = len(region_to_idx)

        # Region query tokens for each region
        self.region_query = nn.Embedding(self.n_regions, config.hidden_size)

    def forward(
            self,
            region_query_seq: np.ndarray,  # (bs, n_queries), str
    ) -> torch.FloatTensor:  # (bs, n_queries, hidden_size)

        region_seq = torch.tensor([[self.region_to_idx[region] for region in regions] for regions in region_query_seq], dtype=torch.long).to(self.region_query.weight.device)
        return self.region_query(region_seq)


class NeuralDecoder(nn.Module):
    def __init__(
            self,
            config: DictConfig,
            region_manager: RegionManager,  # global
            max_F: int,
    ):
        super().__init__()
        self.config = config
        self.region_to_index = region_manager.region_to_index
        self.index_to_region = region_manager.index_to_region
        self.region_query = NeuralRegionQuery(config.query, self.region_to_index)
        self.transformer_blocks = nn.ModuleList(
            [
                NeuralDecoderLayer(
                    d_model=config.transformer.hidden_size,
                    nhead=config.transformer.n_head,
                    dim_feedforward=config.transformer.dim_feedforward,
                    dropout=config.transformer.dropout,
                    bias=config.transformer.bias,
                    activation=config.transformer.activation,
                    norm=config.transformer.norm,
                )
                for _ in range(config.transformer.n_layers)
            ]
        )
        self.out_norm = ScaleNorm(config.transformer.hidden_size ** 0.5) if config.transformer.norm == "scalenorm" else nn.LayerNorm(config.transformer.hidden_size)

        # attention mask
        context_mask = create_context_mask(config.context_forward, config.context_backward, config.embedder.max_F)
        self.register_buffer('context_mask', context_mask, persistent=False)

        # region-wise linear decoder (one linear decoder for each region token)
        self.region_decoders = nn.ModuleDict()
        for region in self.region_to_index.keys():

            self.region_decoders[region] = nn.Linear(config.transformer.hidden_size, max_F)


    def forward(
            self,
            src: torch.FloatTensor,  # (bs, seq_len, hidden_size), from the neural encoder
            region_query_seq: np.ndarray,  # (bs, n_queries), str
    ):




