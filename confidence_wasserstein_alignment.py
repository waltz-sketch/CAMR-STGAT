# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceAwareConditionalWassersteinLoss(nn.Module):
    def __init__(self, num_classes):
        super(ConfidenceAwareConditionalWassersteinLoss, self).__init__()
        self.num_classes = num_classes

    def cosine_distance_matrix(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)

        return 1.0 - torch.matmul(x_norm, y_norm.t())

    def approximate_wasserstein_distance(self, source_feat, target_feat):
        if source_feat.size(0) == 0 or target_feat.size(0) == 0:
            return torch.tensor(0.0, device=source_feat.device)

        cost_matrix = self.cosine_distance_matrix(source_feat, target_feat)

        source_to_target, _ = torch.min(cost_matrix, dim=1)
        target_to_source, _ = torch.min(cost_matrix, dim=0)

        return torch.mean(source_to_target) + torch.mean(target_to_source)

    def forward(self, feat_s, label_s, feat_t, logits_t, threshold=0.7):
        feat_s = feat_s.detach()

        prob_t = F.softmax(logits_t, dim=1)
        conf_t, pseudo_label_t = torch.max(prob_t, dim=1)
        valid_target = conf_t > threshold

        total_loss = 0.0
        valid_classes = 0

        for c in range(self.num_classes):
            source_mask = label_s == c
            target_mask = (pseudo_label_t == c) & valid_target

            source_class_feat = feat_s[source_mask]
            target_class_feat = feat_t[target_mask]

            if source_class_feat.size(0) > 0 and target_class_feat.size(0) > 0:
                total_loss += self.approximate_wasserstein_distance(
                    source_class_feat,
                    target_class_feat,
                )
                valid_classes += 1

        if valid_classes > 0:
            return total_loss / valid_classes

        return torch.tensor(0.0, device=feat_s.device)
