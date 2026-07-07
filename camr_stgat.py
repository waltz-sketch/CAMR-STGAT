# -*- coding: utf-8 -*-
import torch.nn as nn

from adaptive_spectral_graph import AdaptiveMultiResolutionSpectralGraph
from spatio_temporal_graph_attention import ResidualSpatioTemporalGraphAttention


class CAMR_STGAT(nn.Module):
    def __init__(
        self,
        num_classes,
        num_nodes=4,
        node_dim=32,
        hidden_dim=128,
        num_heads=4,
        dropout=0.1,
    ):
        super(CAMR_STGAT, self).__init__()

        self.spectral_graph = AdaptiveMultiResolutionSpectralGraph(
            num_nodes=num_nodes,
            node_dim=node_dim,
        )

        self.stgat = ResidualSpatioTemporalGraphAttention(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.bn = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.band_masks = None

    def forward(self, x64, x128, x256):
        graph_nodes, band_masks = self.spectral_graph(x64, x128, x256)
        feat = self.stgat(graph_nodes)

        self.band_masks = band_masks

        feat = self.bn(feat)
        logits = self.classifier(feat)

        return feat, logits

    def smoothness_loss(self):
        return self.spectral_graph.smoothness_loss()
