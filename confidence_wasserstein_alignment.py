import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceAwareConditionalWassersteinLoss(nn.Module):
    def __init__(self, num_classes, sinkhorn_reg=0.05, sinkhorn_iters=200):
        super().__init__()
        if sinkhorn_reg <= 0 or sinkhorn_iters < 1:
            raise ValueError("Invalid Sinkhorn parameters")
        self.num_classes = num_classes
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iters = sinkhorn_iters

    def cosine_distance_matrix(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return (1.0 - x @ y.transpose(0, 1)).clamp_min(0.0)

    def sinkhorn_transport_plan(self, cost_matrix):
        n, m = cost_matrix.shape
        log_a = cost_matrix.new_full((n,), -math.log(n))
        log_b = cost_matrix.new_full((m,), -math.log(m))
        log_k = -cost_matrix / self.sinkhorn_reg
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)

        for _ in range(self.sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_k + log_v[None, :], dim=1)
            log_v = log_b - torch.logsumexp(log_k + log_u[:, None], dim=0)

        return torch.exp(log_u[:, None] + log_k + log_v[None, :])

    def approximate_wasserstein_distance(self, source_feat, target_feat):
        if source_feat.size(0) == 0 or target_feat.size(0) == 0:
            return (source_feat.sum() + target_feat.sum()) * 0.0
        cost_matrix = self.cosine_distance_matrix(
            source_feat.float(), target_feat.float()
        )
        plan = self.sinkhorn_transport_plan(cost_matrix)
        return (plan * cost_matrix).sum()

    def forward(self, feat_s, label_s, feat_t, logits_t, threshold=0.7):
        probabilities = F.softmax(logits_t, dim=1)
        confidence, pseudo_labels = probabilities.max(dim=1)
        selected = confidence >= threshold
        losses = []

        for cls in range(self.num_classes):
            source_mask = label_s == cls
            target_mask = (pseudo_labels == cls) & selected
            if source_mask.any() and target_mask.any():
                losses.append(
                    self.approximate_wasserstein_distance(
                        feat_s[source_mask], feat_t[target_mask]
                    )
                )

        if not losses:
            return (feat_s.sum() + feat_t.sum()) * 0.0
        return torch.stack(losses).mean()
