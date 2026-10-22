# models/real_knowledge_retriever.py
# -*- coding: utf-8 -*-

"""
真实 Wikidata 知识检索模块。

功能：
    1. 加载 CLIP 编码后的 Wikidata 知识向量；
    2. 使用文本特征进行 text Top-K 知识检索；
    3. 使用图像特征进行 image Top-K 知识检索；
    4. 将检索到的知识向量加权聚合为知识增强特征。

输入文件：
    knowledge/wikidata_knowledge_politifact_embeddings.pt

该 .pt 文件由 scripts/encode_wikidata_knowledge_clip.py 生成，包含：
    embeddings: Tensor[N, 768]
    metadata: List[Dict]
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RealKnowledgeRetriever(nn.Module):
    def __init__(
        self,
        knowledge_embedding_path: str,
        hidden_dim: int = 768,
        text_top_k: int = 5,
        image_top_k: int = 3,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        super().__init__()

        self.knowledge_embedding_path = str(knowledge_embedding_path)
        self.hidden_dim = hidden_dim
        self.text_top_k = text_top_k
        self.image_top_k = image_top_k
        self.temperature = temperature
        self.normalize = normalize

        if self.temperature <= 0:
            raise ValueError("temperature 必须大于 0。")

        path = Path(knowledge_embedding_path)

        if not path.exists():
            raise FileNotFoundError(f"找不到知识向量文件：{path}")

        data = torch.load(path, map_location="cpu")

        if "embeddings" not in data:
            raise KeyError("知识向量文件中缺少 embeddings 字段。")

        embeddings = data["embeddings"].float()

        if embeddings.dim() != 2:
            raise ValueError(f"embeddings 必须是二维张量 [N, D]，当前 shape={tuple(embeddings.shape)}")

        if embeddings.size(1) != hidden_dim:
            raise ValueError(
                f"知识向量维度必须等于 hidden_dim。"
                f"当前 embeddings dim={embeddings.size(1)}, hidden_dim={hidden_dim}"
            )

        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        # persistent=False 表示保存模型 checkpoint 时不重复保存知识库大张量。
        # 加载模型时仍然需要提供 knowledge_embedding_path。
        self.register_buffer(
            "knowledge_embeddings",
            embeddings,
            persistent=False,
        )

        self.metadata: List[Dict] = data.get("metadata", [])

        if len(self.metadata) != embeddings.size(0):
            print(
                "[Warning] metadata 数量和 embeddings 数量不一致："
                f"metadata={len(self.metadata)}, embeddings={embeddings.size(0)}"
            )

    def _safe_top_k(self, k: int) -> int:
        n_knowledge = self.knowledge_embeddings.size(0)

        if n_knowledge <= 0:
            raise ValueError("知识库为空，无法进行 Top-K 检索。")

        return min(k, n_knowledge)

    def _retrieve(
        self,
        query_features: torch.Tensor,
        top_k: int,
    ):
        """
        query_features:
            Tensor[B, D]

        返回：
            knowledge_features: Tensor[B, D]
            top_scores: Tensor[B, K]
            top_indices: Tensor[B, K]
            top_weights: Tensor[B, K]
        """
        if query_features.dim() != 2:
            raise ValueError(f"query_features 必须是 [B, D]，当前 shape={tuple(query_features.shape)}")

        if query_features.size(-1) != self.hidden_dim:
            raise ValueError(
                f"query_features 最后一维必须等于 hidden_dim。"
                f"当前 dim={query_features.size(-1)}, hidden_dim={self.hidden_dim}"
            )

        k = self._safe_top_k(top_k)

        query = query_features

        if self.normalize:
            query = F.normalize(query, dim=-1)

        knowledge = self.knowledge_embeddings

        if self.normalize:
            knowledge = F.normalize(knowledge, dim=-1)

        # similarity: [B, N]
        similarity = torch.matmul(query, knowledge.t())

        top_scores, top_indices = torch.topk(
            similarity,
            k=k,
            dim=-1,
            largest=True,
            sorted=True,
        )

        top_weights = torch.softmax(top_scores / self.temperature, dim=-1)

        # selected_knowledge: [B, K, D]
        selected_knowledge = knowledge[top_indices]

        knowledge_features = torch.sum(
            top_weights.unsqueeze(-1) * selected_knowledge,
            dim=1,
        )

        if self.normalize:
            knowledge_features = F.normalize(knowledge_features, dim=-1)

        return knowledge_features, top_scores, top_indices, top_weights

    def forward(
        self,
        text_query_features: torch.Tensor,
        image_query_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        text_knowledge_features, text_top_scores, text_top_indices, text_top_weights = self._retrieve(
            query_features=text_query_features,
            top_k=self.text_top_k,
        )

        image_knowledge_features, image_top_scores, image_top_indices, image_top_weights = self._retrieve(
            query_features=image_query_features,
            top_k=self.image_top_k,
        )

        knowledge_features = 0.5 * (text_knowledge_features + image_knowledge_features)

        if self.normalize:
            knowledge_features = F.normalize(knowledge_features, dim=-1)

        return {
            "text_knowledge_features": text_knowledge_features,
            "image_knowledge_features": image_knowledge_features,
            "knowledge_features": knowledge_features,
            "text_top_scores": text_top_scores,
            "text_top_indices": text_top_indices,
            "text_top_weights": text_top_weights,
            "image_top_scores": image_top_scores,
            "image_top_indices": image_top_indices,
            "image_top_weights": image_top_weights,
        }

    def get_metadata_by_indices(self, indices: torch.Tensor) -> List[List[Dict]]:
        """
        根据 Top-K indices 返回对应 metadata，方便调试和可解释性分析。

        indices:
            Tensor[B, K]

        返回：
            List[List[Dict]]
        """
        if indices.dim() != 2:
            raise ValueError(f"indices 必须是 [B, K]，当前 shape={tuple(indices.shape)}")

        output = []

        cpu_indices = indices.detach().cpu().tolist()

        for row_indices in cpu_indices:
            row_meta = []

            for idx in row_indices:
                if 0 <= idx < len(self.metadata):
                    row_meta.append(self.metadata[idx])
                else:
                    row_meta.append({})

            output.append(row_meta)

        return output
