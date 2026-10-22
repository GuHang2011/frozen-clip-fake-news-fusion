# models/prompt_learner.py
# -*- coding: utf-8 -*-

"""
作用：
    定义 KICP 中的 Continuous Prompt 模块。

论文对应：
    Continuous Prompt Learning Module

核心思想：
    使用可训练的 continuous prompt vectors 替代人工模板。

论文中的 prompt：
    P = {p1, p2, ..., pnT}

其中：
    nT = 16
    dp = 768

本代码实现：
    1. 创建 16 个可训练 prompt 向量；
    2. 创建 2 个类别向量：
        class 0 = real
        class 1 = fake / rumor
    3. 将 prompt 平均向量加入类别向量；
    4. 输出 class_features，形状为 [2, 768]。

说明：
    这是结构级复现写法。
    论文没有明确说明如何把 continuous prompt 注入 Hugging Face CLIP text encoder，
    所以这里先实现为可训练 prompt + class embedding 的结构级版本。
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousPromptLearner(nn.Module):
    def __init__(
        self,
        class_names: List[str],
        prompt_length: int = 16,
        prompt_dim: int = 768,
        init_std: float = 0.02,
        normalize: bool = True,
    ):
        """
        参数：
            class_names:
                类别名称列表。
                当前二分类任务：
                    ["real", "fake"]

            prompt_length:
                prompt 向量数量。
                KICP 论文设置为 16。

            prompt_dim:
                每个 prompt 向量维度。
                KICP 论文设置为 768。

            init_std:
                prompt 初始化标准差。
                KICP 论文设置为 N(0, 0.02²)。

            normalize:
                是否对输出的 class_features 做 L2 归一化。
                后面 cosine similarity 分类时建议归一化。
        """
        super().__init__()

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.prompt_length = prompt_length
        self.prompt_dim = prompt_dim
        self.init_std = init_std
        self.normalize = normalize

        # ------------------------------------------------------------
        # 1. Continuous Prompt Vectors
        # ------------------------------------------------------------
        # shape: [prompt_length, prompt_dim]
        # 也就是 [16, 768]
        self.prompt_vectors = nn.Parameter(
            torch.empty(prompt_length, prompt_dim)
        )

        # ------------------------------------------------------------
        # 2. Class Embeddings
        # ------------------------------------------------------------
        # shape: [num_classes, prompt_dim]
        # 当前 num_classes = 2，所以是 [2, 768]
        self.class_embeddings = nn.Parameter(
            torch.empty(self.num_classes, prompt_dim)
        )

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化 prompt vectors 和 class embeddings。

        论文说明：
            prompt vectors 从 Gaussian N(0, 0.02²) 初始化。
        """
        nn.init.normal_(
            self.prompt_vectors,
            mean=0.0,
            std=self.init_std,
        )

        nn.init.normal_(
            self.class_embeddings,
            mean=0.0,
            std=self.init_std,
        )

    def forward(self) -> torch.Tensor:
        """
        输出类别特征。

        返回：
            class_features:
                shape = [num_classes, prompt_dim]
                当前是 [2, 768]
        """

        # ------------------------------------------------------------
        # prompt_context:
        #   将 16 个 prompt 向量平均成一个上下文向量。
        #   shape: [1, 768]
        # ------------------------------------------------------------
        prompt_context = self.prompt_vectors.mean(dim=0, keepdim=True)

        # ------------------------------------------------------------
        # class_features:
        #   每个类别 embedding 都加上 prompt_context。
        #   shape: [2, 768]
        # ------------------------------------------------------------
        class_features = self.class_embeddings + prompt_context

        if self.normalize:
            class_features = F.normalize(class_features, dim=-1)

        return class_features

    def extra_repr(self) -> str:
        """
        打印模块信息时显示关键参数。
        """
        return (
            f"num_classes={self.num_classes}, "
            f"prompt_length={self.prompt_length}, "
            f"prompt_dim={self.prompt_dim}, "
            f"init_std={self.init_std}"
        )