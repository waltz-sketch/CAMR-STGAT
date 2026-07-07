# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSpectralNodeGenerator(nn.Module):
    def __init__(self, in_freqs, num_nodes=4, node_dim=32):
        super(AdaptiveSpectralNodeGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.band_assignment = nn.Parameter(torch.randn(num_nodes, in_freqs))
        self.node_projector = nn.Linear(in_freqs, node_dim)
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, x):
        attn_mask = F.softmax(self.band_assignment, dim=1)

        nodes = []
        for i in range(self.num_nodes):
            weighted_spectrum = x * attn_mask[i].unsqueeze(0)
            node_feat = self.node_projector(weighted_spectrum)
            nodes.append(node_feat)

        out_nodes = torch.stack(nodes, dim=1)
        out_nodes = F.relu(self.layer_norm(out_nodes))

        return out_nodes, attn_mask


class AdaptiveMultiResolutionSpectralGraph(nn.Module):
    def __init__(self, num_nodes=4, node_dim=32):
        super(AdaptiveMultiResolutionSpectralGraph, self).__init__()

        self.node_gen_64 = AdaptiveSpectralNodeGenerator(64, num_nodes, node_dim)
        self.node_gen_128 = AdaptiveSpectralNodeGenerator(128, num_nodes, node_dim)
        self.node_gen_256 = AdaptiveSpectralNodeGenerator(256, num_nodes, node_dim)

        self.band_masks = (None, None, None)

    def forward(self, x64, x128, x256):
        batch_size, seq_len, _ = x64.size()

        n64, m64 = self.node_gen_64(x64.reshape(batch_size * seq_len, -1))
        n128, m128 = self.node_gen_128(x128.reshape(batch_size * seq_len, -1))
        n256, m256 = self.node_gen_256(x256.reshape(batch_size * seq_len, -1))

        graph_nodes = torch.cat([n64, n128, n256], dim=1)
        graph_nodes = graph_nodes.reshape(
            batch_size, seq_len, graph_nodes.size(1), graph_nodes.size(2)
        )

        self.band_masks = (m64, m128, m256)

        return graph_nodes, self.band_masks

    def smoothness_loss(self):
        if self.band_masks[0] is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        def total_variation(mask):
            return torch.sum(torch.abs(mask[:, 1:] - mask[:, :-1]))

        return (
            total_variation(self.band_masks[0])
            + total_variation(self.band_masks[1])
            + total_variation(self.band_masks[2])
        )
