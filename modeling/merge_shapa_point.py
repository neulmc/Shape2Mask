import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.GAT import GeometricGAT

class FeatureNormalizer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, features_list):
        return [self.layer_norm(f) for f in features_list]


class TransformerFusion(nn.Module):
    def __init__(self, feature_dim=256, nhead=8, merge_weight = 0.1, num_points=8,
                 fusion_gat = False):
        super().__init__()
        self.norm1 = FeatureNormalizer(feature_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True  # 使用更直观的batch_first模式
        )
        self.weight_proj = nn.Linear(feature_dim, 1)
        self.merge_weight = merge_weight
        nn.init.constant_(self.weight_proj.bias.data, 0.0)
        nn.init.constant_(self.weight_proj.weight.data, 0.0)
        self.fusion_gat = fusion_gat
        if fusion_gat:
            self.gat = GeometricGAT(feature_dim, feature_dim, feature_dim, num_points)

    def forward(self, coords, features):
        # 特征归一化 # 这个 N 是point的数量，也就是8
        if self.fusion_gat:
            features_gat = [self.gat(feature, coord) for (feature, coord) in zip(features, coords)]
            features = [feature + feature_gat for (feature, feature_gat) in zip (features, features_gat)]

        features = self.norm1(features)  # list of [B,N,256]

        # 堆叠特征 [B,N,S,256]
        stacked_feat = torch.stack(features, dim=2)  # S=4（来源数）

        # 合并BN维度 [B*N, S, 256]
        B, N, S, _ = stacked_feat.shape
        trans_feat = stacked_feat.view(B * N, S, -1)

        # Transformer处理 (自动处理序列维度)
        trans_feat = self.transformer(trans_feat)  # [B*N, S, 256]

        # 计算权重 [B*N, S, 1] -> [B,N,S]
        weights = self.weight_proj(trans_feat).softmax(dim=1)
        weights = weights.view(B, N, S)
        weights = weights * self.merge_weight + (1-self.merge_weight) * (1/S)

        # 融合坐标 [B,N,S,2] -> [B,N,2]
        stacked_coords = torch.stack(coords, dim=2)  # [B,N,S,2]
        fused_coords = (stacked_coords * weights.unsqueeze(-1)).sum(dim=2)

        return fused_coords, weights