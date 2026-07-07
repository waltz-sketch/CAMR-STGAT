# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualSpatialGraphAttention(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_heads=4, dropout=0.1):
        super(ResidualSpatialGraphAttention, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, graph_nodes):
        batch_size, seq_len, num_nodes, node_dim = graph_nodes.size()

        x = graph_nodes.reshape(batch_size * seq_len, num_nodes, node_dim)
        x = self.embedding(x)

        try:
            attn_output, attn_weights = self.multihead_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
        except TypeError:
            attn_output, attn_weights = self.multihead_attn(x, x, x)

        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ffn(x))

        z_max = torch.max(x, dim=1)[0]
        z_mean = torch.mean(x, dim=1)
        spatial_feat = z_max + z_mean

        spatial_feat = spatial_feat.reshape(batch_size, seq_len, -1)

        return spatial_feat, attn_weights


class TemporalAttentionReadout(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttentionReadout, self).__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequence_feat):
        attn_scores = self.attention_net(sequence_feat)
        attn_weights = F.softmax(attn_scores, dim=1)
        global_feat = torch.sum(sequence_feat * attn_weights, dim=1)

        return global_feat, attn_weights.squeeze(-1)


class ResidualSpatioTemporalGraphAttention(nn.Module):
    def __init__(self, node_dim=32, hidden_dim=128, num_heads=4, dropout=0.1):
        super(ResidualSpatioTemporalGraphAttention, self).__init__()

        self.spatial_attention = ResidualSpatialGraphAttention(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=dropout,
        )

        self.temporal_attention = TemporalAttentionReadout(hidden_dim)

        self.spatial_attn_weights = None
        self.temporal_attn_weights = None

    def forward(self, graph_nodes):
        spatial_feat, spatial_attn = self.spatial_attention(graph_nodes)

        self.gru.flatten_parameters()
        temporal_feat, _ = self.gru(spatial_feat)

        global_feat, temporal_attn = self.temporal_attention(temporal_feat)

        self.spatial_attn_weights = spatial_attn
        self.temporal_attn_weights = temporal_attn

        return global_feat
