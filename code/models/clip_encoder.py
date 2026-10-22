# models/clip_encoder.py
# -*- coding: utf-8 -*-

"""
作用：
    定义 KICP 项目中的 CLIP 特征提取模块。

输入：
    input_ids
    attention_mask
    pixel_values

输出：
    text_seq_features:
        文本 token 级特征，形状约为 [batch_size, text_length, 768]

    image_seq_features:
        图像 patch 级特征，形状约为 [batch_size, image_tokens, 768]

    text_pooled_features:
        文本全局特征，形状为 [batch_size, 768]

    image_pooled_features:
        图像全局特征，形状为 [batch_size, 768]

说明：
    KICP 论文使用 CLIP ViT-L/14@336px，并冻结 CLIP 参数。
    Hugging Face 对应模型名：
        openai/clip-vit-large-patch14-336
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class KICPCLIPEncoder(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        hidden_dim: int = 768,
        freeze_clip: bool = True,
        normalize_features: bool = True,
    ):
        super().__init__()

        self.clip_model_name = clip_model_name
        self.hidden_dim = hidden_dim
        self.freeze_clip = freeze_clip
        self.normalize_features = normalize_features

        print("正在加载 CLIPModel...")
        print(f"模型名称：{self.clip_model_name}")

        self.clip = CLIPModel.from_pretrained(self.clip_model_name)

        print("CLIPModel 加载完成。")

        text_hidden_size = self.clip.config.text_config.hidden_size
        vision_hidden_size = self.clip.config.vision_config.hidden_size
        projection_dim = self.clip.config.projection_dim

        print("CLIP 配置信息：")
        print(f"text hidden size   : {text_hidden_size}")
        print(f"vision hidden size : {vision_hidden_size}")
        print(f"projection dim     : {projection_dim}")

        if text_hidden_size == hidden_dim:
            self.text_seq_projection = nn.Identity()
        else:
            self.text_seq_projection = nn.Linear(text_hidden_size, hidden_dim)

        if vision_hidden_size == hidden_dim:
            self.image_seq_projection = nn.Identity()
        else:
            self.image_seq_projection = nn.Linear(vision_hidden_size, hidden_dim)

        if projection_dim == hidden_dim:
            self.text_pooled_projection = nn.Identity()
            self.image_pooled_projection = nn.Identity()
        else:
            self.text_pooled_projection = nn.Linear(projection_dim, hidden_dim)
            self.image_pooled_projection = nn.Linear(projection_dim, hidden_dim)

        if self.freeze_clip:
            self.freeze_clip_parameters()

    def freeze_clip_parameters(self):
        for param in self.clip.parameters():
            param.requires_grad = False

        print("已冻结 CLIP 主体参数。")

    def _run_text_model(self, input_ids, attention_mask):
        """
        兼容不同 transformers 版本的 text_model 调用方式。
        """
        try:
            outputs = self.clip.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        except TypeError:
            outputs = self.clip.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs[0]

        if hasattr(outputs, "pooler_output"):
            pooler_output = outputs.pooler_output
        else:
            pooler_output = outputs[1]

        return last_hidden_state, pooler_output

    def _run_vision_model(self, pixel_values):
        """
        兼容不同 transformers 版本的 vision_model 调用方式。
        """
        try:
            outputs = self.clip.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
        except TypeError:
            outputs = self.clip.vision_model(
                pixel_values=pixel_values,
            )

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs[0]

        if hasattr(outputs, "pooler_output"):
            pooler_output = outputs.pooler_output
        else:
            pooler_output = outputs[1]

        return last_hidden_state, pooler_output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # 1. 文本编码
        text_seq_features, text_pooled = self._run_text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_pooled_features = self.clip.text_projection(text_pooled)

        # 2. 图像编码
        image_seq_features, image_pooled = self._run_vision_model(
            pixel_values=pixel_values,
        )

        image_pooled_features = self.clip.visual_projection(image_pooled)

        # 3. token / patch 级特征统一投影到 hidden_dim=768
        text_seq_features = self.text_seq_projection(text_seq_features)
        image_seq_features = self.image_seq_projection(image_seq_features)

        # 4. pooled 特征统一投影到 hidden_dim=768
        text_pooled_features = self.text_pooled_projection(text_pooled_features)
        image_pooled_features = self.image_pooled_projection(image_pooled_features)

        # 5. pooled 特征归一化
        if self.normalize_features:
            text_pooled_features = F.normalize(text_pooled_features, dim=-1)
            image_pooled_features = F.normalize(image_pooled_features, dim=-1)

        return {
            "text_seq_features": text_seq_features,
            "image_seq_features": image_seq_features,
            "text_pooled_features": text_pooled_features,
            "image_pooled_features": image_pooled_features,
        }