# models/knowledge_retriever.py
# -*- coding: utf-8 -*-

"""
作用：
    定义 KICP 项目中的 Knowledge Retriever / Knowledge Enhancement 模块。

当前版本：
    结构级复现版本。

核心思想：
    1. 构建一个可训练的知识向量库 knowledge_bank；
    2. 使用文本全局特征 text_pooled_features 去检索相关知识；
    3. 使用图像全局特征 image_pooled_features 去检索相关知识；
    4. 对 Top-K 知识向量做加权融合；
    5. 输出知识增强特征。

说明：
    论文中知识来源包括：
        - 显式结构化知识，例如 Wikidata；
        - 隐式语义知识，例如视觉语言模型语义关联。

    当前版本先不直接接入 Wikidata，
    而是用可训练 knowledge_bank 模拟知识库向量。
    后面可以把真实 Wikidata 文本编码后替换 knowledge_bank。
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeRetriever(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        knowledge_bank_size: int = 64,
        top_k_text: int = 5,
        top_k_image: int = 5,
        init_std: float = 0.02,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        """
        参数：
            hidden_dim:
                特征维度，KICP 中统一为 768。

            knowledge_bank_size:
                知识库大小。
                当前先用 64 个知识向量模拟。
                后面真实知识库可以扩大。

            top_k_text:
                文本侧检索 Top-K 知识数量。

            top_k_image:
                图像侧检索 Top-K 知识数量。

            init_std:
                知识向量初始化标准差。

            temperature:
                softmax 温度系数。

            normalize:
                是否对查询特征和知识向量做 L2 normalize。
        """
        super().__init__()

        if knowledge_bank_size <= 0:
            raise ValueError("knowledge_bank_size 必须大于 0。")

        if top_k_text <= 0 or top_k_image <= 0:
            raise ValueError("top_k_text 和 top_k_image 必须大于 0。")

        if temperature <= 0:
            raise ValueError("temperature 必须大于 0。")

        self.hidden_dim = hidden_dim
        self.knowledge_bank_size = knowledge_bank_size
        self.top_k_text = top_k_text
        self.top_k_image = top_k_image
        self.init_std = init_std
        self.temperature = temperature
        self.normalize = normalize

        # ------------------------------------------------------------
        # 可训练知识库
        # shape = [knowledge_bank_size, hidden_dim]
        # ------------------------------------------------------------
        self.knowledge_bank = nn.Parameter(
            torch.empty(knowledge_bank_size, hidden_dim)
        )

        # 文本知识输出投影
        self.text_knowledge_projection = nn.Linear(hidden_dim, hidden_dim)

        # 图像知识输出投影
        self.image_knowledge_projection = nn.Linear(hidden_dim, hidden_dim)

        # 文本知识 + 图像知识 融合
        self.fusion_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        self.norm_text = nn.LayerNorm(hidden_dim)
        self.norm_image = nn.LayerNorm(hidden_dim)
        self.norm_fusion = nn.LayerNorm(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(
            self.knowledge_bank,
            mean=0.0,
            std=self.init_std,
        )

    def retrieve(
        self,
        query_features: torch.Tensor,
        top_k: int,
    ) -> Dict[str, torch.Tensor]:
        """
        根据 query_features 从 knowledge_bank 中检索 Top-K 知识。

        参数：
            query_features:
                shape = [B, hidden_dim]

            top_k:
                检索前 K 个知识向量。

        返回：
            retrieved_features:
                shape = [B, hidden_dim]

            topk_indices:
                shape = [B, top_k]

            topk_scores:
                shape = [B, top_k]

            topk_weights:
                shape = [B, top_k]
        """
        if query_features.dim() != 2:
            raise ValueError(
                f"query_features 必须是 [B, D]，当前 shape={query_features.shape}"
            )

        if query_features.size(-1) != self.hidden_dim:
            raise ValueError(
                f"query_features 最后一维必须是 {self.hidden_dim}，当前是 {query_features.size(-1)}"
            )

        actual_top_k = min(top_k, self.knowledge_bank_size)

        if self.normalize:
            query = F.normalize(query_features, dim=-1)
            bank = F.normalize(self.knowledge_bank, dim=-1)
        else:
            query = query_features
            bank = self.knowledge_bank

        # similarity: [B, knowledge_bank_size]
        similarity = torch.matmul(query, bank.t())

        # Top-K
        topk_scores, topk_indices = torch.topk(
            similarity,
            k=actual_top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )

        # selected_bank: [B, top_k, hidden_dim]
        selected_bank = bank[topk_indices]

        # 权重: [B, top_k]
        topk_weights = torch.softmax(
            topk_scores / self.temperature,
            dim=-1,
        )

        # [B, top_k] -> [B, top_k, 1]
        weights = topk_weights.unsqueeze(-1)

        # 加权求和: [B, hidden_dim]
        retrieved_features = torch.sum(
            selected_bank * weights,
            dim=1,
        )

        return {
            "retrieved_features": retrieved_features,
            "topk_indices": topk_indices,
            "topk_scores": topk_scores,
            "topk_weights": topk_weights,
        }

    def forward(
        self,
        text_pooled_features: torch.Tensor,
        image_pooled_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        输入：
            text_pooled_features:
                [B, 768]

            image_pooled_features:
                [B, 768]

        输出：
            text_knowledge_features:
                [B, 768]

            image_knowledge_features:
                [B, 768]

            knowledge_features:
                [B, 768]
        """

        text_result = self.retrieve(
            query_features=text_pooled_features,
            top_k=self.top_k_text,
        )

        image_result = self.retrieve(
            query_features=image_pooled_features,
            top_k=self.top_k_image,
        )

        text_knowledge_features = text_result["retrieved_features"]
        image_knowledge_features = image_result["retrieved_features"]

        text_knowledge_features = self.text_knowledge_projection(
            text_knowledge_features
        )
        image_knowledge_features = self.image_knowledge_projection(
            image_knowledge_features
        )

        text_knowledge_features = self.norm_text(text_knowledge_features)
        image_knowledge_features = self.norm_image(image_knowledge_features)

        concat_features = torch.cat(
            [text_knowledge_features, image_knowledge_features],
            dim=-1,
        )

        knowledge_features = self.fusion_projection(concat_features)
        knowledge_features = self.norm_fusion(knowledge_features)

        return {
            "text_knowledge_features": text_knowledge_features,
            "image_knowledge_features": image_knowledge_features,
            "knowledge_features": knowledge_features,
            "text_topk_indices": text_result["topk_indices"],
            "image_topk_indices": image_result["topk_indices"],
            "text_topk_scores": text_result["topk_scores"],
            "image_topk_scores": image_result["topk_scores"],
            "text_topk_weights": text_result["topk_weights"],
            "image_topk_weights": image_result["topk_weights"],
        }

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"knowledge_bank_size={self.knowledge_bank_size}, "
            f"top_k_text={self.top_k_text}, "
            f"top_k_image={self.top_k_image}, "
            f"temperature={self.temperature}"
        )