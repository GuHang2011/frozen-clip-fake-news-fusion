# models/co_attention.py
# -*- coding: utf-8 -*-

"""
作用：
    定义 KICP 项目中的 Co-Attention 模块。

输入：
    a_features:
        第一种模态特征，例如图像特征
        shape = [batch_size, seq_len_a, hidden_dim]

    b_features:
        第二种模态特征，例如文本特征
        shape = [batch_size, seq_len_b, hidden_dim]

输出：
    fused_features:
        融合后的特征
        shape = [batch_size, hidden_dim]

KICP 论文中的 Co-Attention 思路：
    H_ab  = a + MultiHead(a, b, b)
    H'_ab = FFN(H_ab)

    H_ba  = b + MultiHead(b, a, a)
    H'_ba = FFN(H_ba)

    Ca(a,b) = concat(H'_ab, H'_ba) W

说明：
    文本长度和图像 patch 数量通常不一样。
    例如：
        text_seq_features  = [B, 23, 768]
        image_seq_features = [B, 577, 768]

    因此这里先分别做 attention，再做 mean pooling，
    最后 concat 成 [B, 1536]，再投影回 [B, 768]。
"""

from typing import Optional, Dict

import torch
import torch.nn as nn


class CoAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_dim: int = 2048,
    ):
        """
        参数：
            hidden_dim:
                输入和输出特征维度。
                KICP 中统一使用 768。

            num_heads:
                多头注意力头数。
                KICP 方法部分中：
                    Stage 1 可用 4 heads
                    Stage 2 可用 4 heads
                    Stage 3 可用 8 heads

            dropout:
                dropout 概率。

            ffn_dim:
                FFN 中间层维度。
                这里是工程实现参数，后面可在配置中调整。
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim 必须能被 num_heads 整除。"
                f"当前 hidden_dim={hidden_dim}, num_heads={num_heads}"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffn_dim = ffn_dim

        # ------------------------------------------------------------
        # a attend to b:
        # query = a, key = b, value = b
        # ------------------------------------------------------------
        self.attn_ab = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ------------------------------------------------------------
        # b attend to a:
        # query = b, key = a, value = a
        # ------------------------------------------------------------
        self.attn_ba = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN for H_ab
        self.ffn_ab = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

        # FFN for H_ba
        self.ffn_ba = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

        # LayerNorm 用于稳定训练
        self.norm_ab_1 = nn.LayerNorm(hidden_dim)
        self.norm_ab_2 = nn.LayerNorm(hidden_dim)

        self.norm_ba_1 = nn.LayerNorm(hidden_dim)
        self.norm_ba_2 = nn.LayerNorm(hidden_dim)

        # concat 后投影：
        # [B, 768] + [B, 768] -> [B, 1536] -> [B, 768]
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def masked_mean_pooling(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对序列特征做 mean pooling。

        参数：
            features:
                shape = [B, L, D]

            padding_mask:
                shape = [B, L]
                True 表示当前位置是 padding，需要忽略。
                None 表示没有 padding，直接平均。

        返回：
            pooled:
                shape = [B, D]
        """
        if padding_mask is None:
            return features.mean(dim=1)

        # padding_mask: True = padding
        # valid_mask: True = 有效 token
        valid_mask = (~padding_mask).float()

        # [B, L] -> [B, L, 1]
        valid_mask = valid_mask.unsqueeze(-1)

        features = features * valid_mask

        denom = valid_mask.sum(dim=1).clamp(min=1.0)

        pooled = features.sum(dim=1) / denom

        return pooled

    def forward(
        self,
        a_features: torch.Tensor,
        b_features: torch.Tensor,
        a_padding_mask: Optional[torch.Tensor] = None,
        b_padding_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        """
        前向传播。

        参数：
            a_features:
                shape = [B, seq_len_a, hidden_dim]

            b_features:
                shape = [B, seq_len_b, hidden_dim]

            a_padding_mask:
                shape = [B, seq_len_a]
                True 表示 padding。

            b_padding_mask:
                shape = [B, seq_len_b]
                True 表示 padding。

            return_details:
                是否返回中间结果。

        返回：
            默认返回 fused_features:
                shape = [B, hidden_dim]

            如果 return_details=True，返回 dict。
        """

        if a_features.dim() != 3:
            raise ValueError(
                f"a_features 必须是 3 维 [B, L, D]，当前 shape={a_features.shape}"
            )

        if b_features.dim() != 3:
            raise ValueError(
                f"b_features 必须是 3 维 [B, L, D]，当前 shape={b_features.shape}"
            )

        if a_features.size(-1) != self.hidden_dim:
            raise ValueError(
                f"a_features 最后一维必须是 {self.hidden_dim}，当前是 {a_features.size(-1)}"
            )

        if b_features.size(-1) != self.hidden_dim:
            raise ValueError(
                f"b_features 最后一维必须是 {self.hidden_dim}，当前是 {b_features.size(-1)}"
            )

        # ------------------------------------------------------------
        # 1. a attends to b
        # H_ab = a + MultiHead(a, b, b)
        # ------------------------------------------------------------
        attn_ab_output, attn_ab_weights = self.attn_ab(
            query=a_features,
            key=b_features,
            value=b_features,
            key_padding_mask=b_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        h_ab = self.norm_ab_1(a_features + attn_ab_output)

        # H'_ab = FFN(H_ab)
        h_ab_ffn = self.ffn_ab(h_ab)
        h_ab = self.norm_ab_2(h_ab + h_ab_ffn)

        # ------------------------------------------------------------
        # 2. b attends to a
        # H_ba = b + MultiHead(b, a, a)
        # ------------------------------------------------------------
        attn_ba_output, attn_ba_weights = self.attn_ba(
            query=b_features,
            key=a_features,
            value=a_features,
            key_padding_mask=a_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        h_ba = self.norm_ba_1(b_features + attn_ba_output)

        # H'_ba = FFN(H_ba)
        h_ba_ffn = self.ffn_ba(h_ba)
        h_ba = self.norm_ba_2(h_ba + h_ba_ffn)

        # ------------------------------------------------------------
        # 3. Pooling
        # 因为 a 和 b 序列长度不一样，先各自池化成 [B, 768]
        # ------------------------------------------------------------
        h_ab_pooled = self.masked_mean_pooling(
            h_ab,
            padding_mask=a_padding_mask,
        )

        h_ba_pooled = self.masked_mean_pooling(
            h_ba,
            padding_mask=b_padding_mask,
        )

        # ------------------------------------------------------------
        # 4. Concatenate and project
        # [B, 768] + [B, 768] -> [B, 1536] -> [B, 768]
        # ------------------------------------------------------------
        concat_features = torch.cat(
            [h_ab_pooled, h_ba_pooled],
            dim=-1,
        )

        fused_features = self.output_projection(concat_features)

        if return_details:
            return {
                "fused_features": fused_features,
                "h_ab": h_ab,
                "h_ba": h_ba,
                "h_ab_pooled": h_ab_pooled,
                "h_ba_pooled": h_ba_pooled,
                "attn_ab_weights": attn_ab_weights,
                "attn_ba_weights": attn_ba_weights,
            }

        return fused_features

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"dropout={self.dropout}, "
            f"ffn_dim={self.ffn_dim}"
        )