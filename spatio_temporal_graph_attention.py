import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualSpatialGraphAttention(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, graph_nodes):
        batch_size, seq_len, num_nodes, node_dim = graph_nodes.shape
        x = graph_nodes.reshape(batch_size * seq_len, num_nodes, node_dim)
        x = self.embedding(x)
        attended, weights = self.multihead_attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        x = self.norm(x + attended)
        spatial_feat = torch.cat((x.mean(dim=1), x.amax(dim=1)), dim=-1)
        return spatial_feat.reshape(batch_size, seq_len, -1), weights


class TemporalAttentionReadout(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequence_feat):
        weights = F.softmax(self.attention_net(sequence_feat), dim=1)
        global_feat = (sequence_feat * weights).sum(dim=1)
        return global_feat, weights.squeeze(-1)


class ResidualSpatioTemporalGraphAttention(nn.Module):
    def __init__(self, node_dim=32, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.spatial_attention = ResidualSpatialGraphAttention(
            node_dim, hidden_dim, num_heads, dropout
        )
        self.gru = nn.GRU(
            input_size=2 * hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_attention = TemporalAttentionReadout(hidden_dim)
        self.spatial_attn_weights = None
        self.temporal_attn_weights = None

    def forward(self, graph_nodes):
        spatial_feat, spatial_weights = self.spatial_attention(graph_nodes)
        temporal_feat, _ = self.gru(spatial_feat)
        global_feat, temporal_weights = self.temporal_attention(temporal_feat)
        self.spatial_attn_weights = spatial_weights
        self.temporal_attn_weights = temporal_weights
        return global_feat
