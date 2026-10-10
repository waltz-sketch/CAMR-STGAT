import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSpectralNodeGenerator(nn.Module):
    def __init__(self, in_freqs, num_nodes=4, node_dim=32):
        super().__init__()
        self.band_assignment = nn.Parameter(torch.randn(num_nodes, in_freqs))
        self.node_projector = nn.Linear(1, node_dim)
        self.layer_norm = nn.LayerNorm(node_dim)

    def get_band_mask(self):
        return F.softmax(self.band_assignment, dim=0)

    def forward(self, x):
        mask = self.get_band_mask()
        response = torch.matmul(x, mask.transpose(0, 1))
        nodes = F.relu(self.layer_norm(self.node_projector(response.unsqueeze(-1))))
        return nodes, mask

    def smoothness_loss(self):
        mask = self.get_band_mask()
        return (mask[:, 1:] - mask[:, :-1]).square().sum()


class AdaptiveMultiResolutionSpectralGraph(nn.Module):
    def __init__(self, num_nodes=4, node_dim=32):
        super().__init__()
        self.node_gen_64 = AdaptiveSpectralNodeGenerator(64, num_nodes, node_dim)
        self.node_gen_128 = AdaptiveSpectralNodeGenerator(128, num_nodes, node_dim)
        self.node_gen_256 = AdaptiveSpectralNodeGenerator(256, num_nodes, node_dim)
        self.band_masks = None

    def forward(self, x64, x128, x256):
        batch_size, seq_len, _ = x64.shape
        n64, m64 = self.node_gen_64(x64.reshape(-1, 64))
        n128, m128 = self.node_gen_128(x128.reshape(-1, 128))
        n256, m256 = self.node_gen_256(x256.reshape(-1, 256))
        graph_nodes = torch.cat((n64, n128, n256), dim=1)
        graph_nodes = graph_nodes.reshape(batch_size, seq_len, -1, graph_nodes.size(-1))
        self.band_masks = (m64, m128, m256)
        return graph_nodes, self.band_masks

    def smoothness_loss(self):
        return (
            self.node_gen_64.smoothness_loss()
            + self.node_gen_128.smoothness_loss()
            + self.node_gen_256.smoothness_loss()
        )
