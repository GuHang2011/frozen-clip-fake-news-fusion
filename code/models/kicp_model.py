# models/kicp_model.py
# -*- coding: utf-8 -*-

"""
KICP 主模型：真实 Wikidata 知识库兼容版 V2。

修复内容：
    1. 兼容 KICPCLIPEncoder 的不同参数名；
    2. 兼容 ContinuousPromptLearner 的不同参数名；
    3. 兼容 CoAttentionBlock 的不同 forward 参数形式；
    4. 支持真实 Wikidata 知识向量检索：
        knowledge_embedding_path="knowledge/wikidata_knowledge_politifact_embeddings.pt"

知识模式：
    - knowledge_embedding_path=None:
        使用原来的模拟可训练 knowledge_bank。
    - knowledge_embedding_path 不为空:
        使用真实 Wikidata CLIP 知识向量。
"""

from typing import Dict, Optional, List
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip_encoder import KICPCLIPEncoder
from models.co_attention import CoAttentionBlock
from models.prompt_learner import ContinuousPromptLearner
from models.knowledge_retriever import KnowledgeRetriever
from models.real_knowledge_retriever import RealKnowledgeRetriever


class FallbackContinuousPromptLearner(nn.Module):
    """
    兜底版连续提示学习器。
    """

    def __init__(
        self,
        class_names: List[str],
        hidden_dim: int = 768,
        prompt_length: int = 16,
        init_std: float = 0.02,
        normalize: bool = True,
    ):
        super().__init__()

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.hidden_dim = hidden_dim
        self.prompt_length = prompt_length
        self.normalize = normalize

        self.prompt_vectors = nn.Parameter(
            torch.empty(prompt_length, hidden_dim)
        )

        self.class_embeddings = nn.Parameter(
            torch.empty(self.num_classes, hidden_dim)
        )

        nn.init.normal_(self.prompt_vectors, mean=0.0, std=init_std)
        nn.init.normal_(self.class_embeddings, mean=0.0, std=init_std)

    def forward(self) -> torch.Tensor:
        prompt_context = self.prompt_vectors.mean(dim=0, keepdim=True)
        class_features = self.class_embeddings + prompt_context

        if self.normalize:
            class_features = F.normalize(class_features, dim=-1)

        return class_features


def _filter_kwargs_for_signature(cls, kwargs: Dict) -> Dict:
    """
    根据类的 __init__ 签名过滤 kwargs，避免 unexpected keyword argument。
    """
    signature = inspect.signature(cls.__init__)
    params = signature.parameters

    for param in params.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs

    return {key: value for key, value in kwargs.items() if key in params}


def build_clip_encoder_compatible(
    clip_model_name: str,
    hidden_dim: int,
    freeze_clip: bool,
) -> nn.Module:
    """
    兼容不同 KICPCLIPEncoder 参数名。
    """
    candidates = [
        {
            "clip_model_name": clip_model_name,
            "hidden_dim": hidden_dim,
            "freeze_clip": freeze_clip,
        },
        {
            "model_name": clip_model_name,
            "hidden_dim": hidden_dim,
            "freeze_clip": freeze_clip,
        },
        {
            "model_name_or_path": clip_model_name,
            "hidden_dim": hidden_dim,
            "freeze_clip": freeze_clip,
        },
    ]

    errors = []

    for kwargs in candidates:
        try:
            filtered_kwargs = _filter_kwargs_for_signature(KICPCLIPEncoder, kwargs)
            return KICPCLIPEncoder(**filtered_kwargs)
        except TypeError as e:
            errors.append(str(e))

    raise TypeError(
        "无法构建 KICPCLIPEncoder，请检查 models/clip_encoder.py 的 __init__ 参数。"
        f"尝试错误：{errors}"
    )


def build_prompt_learner_compatible(
    class_names: List[str],
    hidden_dim: int,
    prompt_length: int,
    init_std: float,
    normalize: bool,
) -> nn.Module:
    """
    兼容不同 ContinuousPromptLearner 参数名。
    """
    num_classes = len(class_names)

    candidates = [
        {
            "class_names": class_names,
            "hidden_dim": hidden_dim,
            "prompt_length": prompt_length,
            "init_std": init_std,
            "normalize": normalize,
        },
        {
            "class_names": class_names,
            "embed_dim": hidden_dim,
            "prompt_length": prompt_length,
            "init_std": init_std,
            "normalize": normalize,
        },
        {
            "class_names": class_names,
            "embedding_dim": hidden_dim,
            "prompt_length": prompt_length,
            "init_std": init_std,
            "normalize": normalize,
        },
        {
            "class_names": class_names,
            "d_model": hidden_dim,
            "prompt_length": prompt_length,
            "init_std": init_std,
            "normalize": normalize,
        },
        {
            "num_classes": num_classes,
            "hidden_dim": hidden_dim,
            "prompt_length": prompt_length,
            "init_std": init_std,
            "normalize": normalize,
        },
        {
            "num_classes": num_classes,
            "embed_dim": hidden_dim,
            "prompt_length": prompt_length,
            "init_std": init_std,
            "normalize": normalize,
        },
        {
            "num_classes": num_classes,
            "embedding_dim": hidden_dim,
            "prompt_length": prompt_length,
            "init_std": init_std,
            "normalize": normalize,
        },
        {
            "num_classes": num_classes,
            "d_model": hidden_dim,
            "prompt_length": prompt_length,
            "init_std": init_std,
            "normalize": normalize,
        },
    ]

    errors = []

    for kwargs in candidates:
        try:
            filtered_kwargs = _filter_kwargs_for_signature(ContinuousPromptLearner, kwargs)

            if len(filtered_kwargs) == 0:
                continue

            return ContinuousPromptLearner(**filtered_kwargs)

        except TypeError as e:
            errors.append(str(e))

    print("[Warning] ContinuousPromptLearner 参数不兼容，已启用 FallbackContinuousPromptLearner。")
    if errors:
        print("[Warning] 原始错误：", errors[-1])

    return FallbackContinuousPromptLearner(
        class_names=class_names,
        hidden_dim=hidden_dim,
        prompt_length=prompt_length,
        init_std=init_std,
        normalize=normalize,
    )


def extract_tensor_from_coattention_output(output):
    """
    兼容不同 CoAttentionBlock 的输出形式：
        - Tensor[B, D]
        - tuple/list，第一个元素是 Tensor
        - dict，优先取 fused_features / output / features
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise ValueError("CoAttentionBlock 返回了空 tuple/list。")

        first = output[0]

        if torch.is_tensor(first):
            return first

        raise ValueError(f"CoAttentionBlock tuple/list 第一个元素不是 Tensor：{type(first)}")

    if isinstance(output, dict):
        for key in [
            "fused_features",
            "multimodal_features",
            "features",
            "output",
            "pooled",
        ]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]

        raise ValueError(f"CoAttentionBlock 返回 dict，但没有找到可用 Tensor。keys={list(output.keys())}")

    raise ValueError(f"CoAttentionBlock 返回类型不支持：{type(output)}")


def run_co_attention_compatible(
    co_attention: nn.Module,
    image_seq_features: torch.Tensor,
    text_seq_features: torch.Tensor,
    text_padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    兼容不同版本 CoAttentionBlock.forward：

    可能形式：
        1. forward(a, b, key_padding_mask_a=None, key_padding_mask_b=None)
        2. forward(a, b, None, text_padding_mask)
        3. forward(a, b, text_padding_mask)
        4. forward(a, b)
    """
    attempts = []

    # 1. 关键词版本
    try:
        output = co_attention(
            image_seq_features,
            text_seq_features,
            key_padding_mask_a=None,
            key_padding_mask_b=text_padding_mask,
        )
        return extract_tensor_from_coattention_output(output)
    except TypeError as e:
        attempts.append(f"keyword_mask_error={e}")

    # 2. 四个位置参数版本
    try:
        output = co_attention(
            image_seq_features,
            text_seq_features,
            None,
            text_padding_mask,
        )
        return extract_tensor_from_coattention_output(output)
    except TypeError as e:
        attempts.append(f"four_args_error={e}")

    # 3. 三个位置参数版本
    try:
        output = co_attention(
            image_seq_features,
            text_seq_features,
            text_padding_mask,
        )
        return extract_tensor_from_coattention_output(output)
    except TypeError as e:
        attempts.append(f"three_args_error={e}")

    # 4. 两个位置参数版本
    try:
        output = co_attention(
            image_seq_features,
            text_seq_features,
        )
        return extract_tensor_from_coattention_output(output)
    except TypeError as e:
        attempts.append(f"two_args_error={e}")

    raise TypeError(
        "无法调用 CoAttentionBlock.forward，请检查 models/co_attention.py 的 forward 参数。"
        f"尝试错误：{attempts}"
    )


class KICPModel(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        class_names: Optional[List[str]] = None,
        hidden_dim: int = 768,
        prompt_length: int = 16,
        prompt_init_std: float = 0.02,
        coattention_heads: int = 4,
        coattention_dropout: float = 0.1,
        coattention_ffn_dim: int = 2048,
        classifier_temperature: float = 0.07,
        freeze_clip: bool = True,
        use_knowledge: bool = True,
        knowledge_bank_size: int = 64,
        top_k_text: int = 5,
        top_k_image: int = 5,
        knowledge_embedding_path: Optional[str] = None,
    ):
        super().__init__()

        if class_names is None:
            class_names = ["real", "fake"]

        if classifier_temperature <= 0:
            raise ValueError("classifier_temperature 必须大于 0。")

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.hidden_dim = hidden_dim
        self.classifier_temperature = classifier_temperature
        self.use_knowledge = use_knowledge
        self.knowledge_embedding_path = knowledge_embedding_path

        if self.use_knowledge and self.knowledge_embedding_path is not None:
            self.knowledge_mode = "real_wikidata"
        elif self.use_knowledge:
            self.knowledge_mode = "simulated_trainable"
        else:
            self.knowledge_mode = "none"

        self.clip_encoder = build_clip_encoder_compatible(
            clip_model_name=clip_model_name,
            hidden_dim=hidden_dim,
            freeze_clip=freeze_clip,
        )

        self.co_attention = CoAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=coattention_heads,
            dropout=coattention_dropout,
            ffn_dim=coattention_ffn_dim,
        )

        if self.use_knowledge:
            if self.knowledge_embedding_path is not None:
                self.knowledge_retriever = RealKnowledgeRetriever(
                    knowledge_embedding_path=self.knowledge_embedding_path,
                    hidden_dim=hidden_dim,
                    text_top_k=top_k_text,
                    image_top_k=top_k_image,
                    temperature=classifier_temperature,
                    normalize=True,
                )
            else:
                self.knowledge_retriever = KnowledgeRetriever(
                    hidden_dim=hidden_dim,
                    knowledge_bank_size=knowledge_bank_size,
                    top_k_text=top_k_text,
                    top_k_image=top_k_image,
                    init_std=0.02,
                    temperature=classifier_temperature,
                    normalize=True,
                )

            self.knowledge_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(coattention_dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.knowledge_fusion_norm = nn.LayerNorm(hidden_dim)

        self.prompt_learner = build_prompt_learner_compatible(
            class_names=class_names,
            hidden_dim=hidden_dim,
            prompt_length=prompt_length,
            init_std=prompt_init_std,
            normalize=True,
        )

        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(coattention_dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        clip_outputs = self.clip_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        text_seq_features = clip_outputs["text_seq_features"]
        image_seq_features = clip_outputs["image_seq_features"]
        text_pooled_features = clip_outputs["text_pooled_features"]
        image_pooled_features = clip_outputs["image_pooled_features"]

        text_padding_mask = attention_mask == 0

        multimodal_features = run_co_attention_compatible(
            co_attention=self.co_attention,
            image_seq_features=image_seq_features,
            text_seq_features=text_seq_features,
            text_padding_mask=text_padding_mask,
        )

        multimodal_features = self.dropout(
            self.fusion_norm(multimodal_features)
        )

        knowledge_outputs = None

        if self.use_knowledge:
            knowledge_outputs = self.knowledge_retriever(
                text_query_features=text_pooled_features,
                image_query_features=image_pooled_features,
            )

            knowledge_features = knowledge_outputs["knowledge_features"]

            knowledge_enhanced_features = self.knowledge_fusion(
                torch.cat([multimodal_features, knowledge_features], dim=-1)
            )

            final_features = self.knowledge_fusion_norm(
                multimodal_features + knowledge_enhanced_features
            )

        else:
            final_features = multimodal_features

        class_features = self.prompt_learner()

        if isinstance(class_features, tuple):
            class_features = class_features[0]

        if isinstance(class_features, dict):
            for key in ["class_features", "features", "output"]:
                if key in class_features and torch.is_tensor(class_features[key]):
                    class_features = class_features[key]
                    break

        if not torch.is_tensor(class_features):
            raise ValueError(f"prompt_learner 输出不是 Tensor：{type(class_features)}")

        if class_features.dim() != 2:
            raise ValueError(
                f"prompt_learner 输出必须是二维张量 [num_classes, hidden_dim]，"
                f"当前 shape={tuple(class_features.shape)}"
            )

        logits = torch.matmul(
            F.normalize(final_features, dim=-1),
            F.normalize(class_features, dim=-1).t(),
        ) / self.classifier_temperature

        probs = torch.softmax(logits, dim=-1)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": probs,
        }

        if labels is not None:
            outputs["loss"] = F.cross_entropy(logits, labels)

        if return_features:
            outputs.update(
                {
                    "final_features": final_features,
                    "multimodal_features": multimodal_features,
                    "class_features": class_features,
                    "text_seq_features": text_seq_features,
                    "image_seq_features": image_seq_features,
                    "text_pooled_features": text_pooled_features,
                    "image_pooled_features": image_pooled_features,
                }
            )

            if knowledge_outputs is not None:
                outputs.update(knowledge_outputs)

        return outputs
