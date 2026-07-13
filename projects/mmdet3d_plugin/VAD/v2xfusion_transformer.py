# ------------------------------------------------------
#  Simple Transformer Decoder for V2X Fusion
#  Inputs:
#     - agent_queries:   [B, Nq, C]
#     - bev_queries:     [B, Nb, C]
#  Output:
#     - outputs: [L, B, Nq, C] per-layer decoder outputs
# ------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------
# Transformer Decoder Layer
# ------------------------------------------------------
class V2XDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 1. Self-attention between agent queries (query <-> query)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Cross-attention: agent queries attend to BEV features
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 3. Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, agent_query, bev_query, bev_pos=None):
        """
        agent_query:  [B, Nq, C]
        bev_query:    [B, Nb, C]
        bev_pos:      [B, Nb, C] or None
        """

        # ---- SELF ATTENTION ----
        q = k = agent_query
        attn_output, _ = self.self_attn(q, k, agent_query)
        agent_query = self.norm1(agent_query + self.dropout1(attn_output))

        # ---- CROSS ATTENTION (agent → BEV attention) ----
        if bev_pos is not None:
            bev_query = bev_query + bev_pos

        attn_output, _ = self.cross_attn(agent_query, bev_query, bev_query)
        agent_query = self.norm2(agent_query + self.dropout2(attn_output))

        # ---- FFN ----
        ffn_out = self.linear2(self.dropout3(self.activation(self.linear1(agent_query))))
        agent_query = self.norm3(agent_query + ffn_out)

        return agent_query


# ------------------------------------------------------
# Full Transformer Decoder (Multiple Layers)
# ------------------------------------------------------
class V2XTransformerDecoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, num_layers=6, ffn_dim=2048, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            V2XDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

    def forward(self, agent_queries, bev_queries, bev_pos=None):
        """
        agent_queries:  [B, Nq, C]
        bev_queries:    [B, Nb, C]
        bev_pos:        [B, Nb, C] or None

        Returns:
            outputs: [L, B, Nq, C]
        """

        out_per_layer = []

        query = agent_queries

        for layer in self.layers:
            query = layer(query, bev_queries, bev_pos)
            out_per_layer.append(query)

        # shape: [num_layers, B, Nq, C]
        return torch.stack(out_per_layer, dim=0)
