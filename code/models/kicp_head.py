# models/kicp_head.py
# -*- coding: utf-8 -*-
#
# KICP head that runs on PRECOMPUTED frozen-CLIP features (see
# scripts/precompute_clip.py). It reproduces models/kicp_model.py exactly
# (minus the frozen CLIP body) and adds ablation switches for Table 3:
#
#   use_image       : -IM   when False  (drop the image modality)
#   use_text        : -TX   when False  (drop the text modality)
#   use_coattention : -CoAttention when False (mean-pool + linear fusion)
#   use_prompt      : -CPT  when False  (plain linear classifier head)
#   use_knowledge   : -KB   when False  (no Wikidata knowledge enhancement)
#
# The trainable image patch projection 1024->768 lives here (it is NOT part of
# the frozen CLIP body in the original model). ASCII only.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.co_attention import CoAttentionBlock
from models.prompt_learner import ContinuousPromptLearner
from models.real_knowledge_retriever import RealKnowledgeRetriever


def masked_mean(features, padding_mask):
    # features [B,L,D]; padding_mask [B,L] True=pad
    if padding_mask is None:
        return features.mean(dim=1)
    valid = (~padding_mask).float().unsqueeze(-1)
    return (features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


class KICPHead(nn.Module):
    def __init__(
        self,
        knowledge_embedding_path: Optional[str] = None,
        hidden_dim: int = 768,
        image_in_dim: int = 1024,
        prompt_length: int = 16,
        coattention_heads: int = 8,
        coattention_ffn_dim: int = 2048,
        dropout: float = 0.6,
        classifier_temperature: float = 0.07,
        top_k_text: int = 5,
        top_k_image: int = 5,
        use_image: bool = True,
        use_text: bool = True,
        use_coattention: bool = True,
        use_prompt: bool = True,
        use_knowledge: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.classifier_temperature = classifier_temperature
        self.use_image = use_image
        self.use_text = use_text
        self.use_coattention = use_coattention
        self.use_prompt = use_prompt
        self.use_knowledge = use_knowledge

        # trainable image patch projection 1024 -> 768 (from the encoder)
        self.image_seq_projection = nn.Linear(image_in_dim, hidden_dim)

        if use_coattention:
            self.co_attention = CoAttentionBlock(
                hidden_dim=hidden_dim, num_heads=coattention_heads,
                dropout=dropout, ffn_dim=coattention_ffn_dim,
            )
        else:
            self.simple_fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if use_knowledge:
            if knowledge_embedding_path is None:
                raise ValueError("use_knowledge=True requires knowledge_embedding_path")
            self.knowledge_retriever = RealKnowledgeRetriever(
                knowledge_embedding_path=knowledge_embedding_path,
                hidden_dim=hidden_dim, text_top_k=top_k_text,
                image_top_k=top_k_image, temperature=classifier_temperature,
                normalize=True,
            )
            self.knowledge_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
            )
            self.knowledge_fusion_norm = nn.LayerNorm(hidden_dim)

        if use_prompt:
            self.prompt_learner = ContinuousPromptLearner(
                class_names=["real", "fake"], prompt_length=prompt_length,
                prompt_dim=hidden_dim, init_std=0.02, normalize=True,
            )
        else:
            self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text_seq, image_seq_raw, text_pooled, image_pooled, attn_mask, labels=None):
        # image patch projection
        image_seq = self.image_seq_projection(image_seq_raw)  # [B,577,768]

        if not self.use_image:
            image_seq = torch.zeros_like(image_seq)
            image_pooled = torch.zeros_like(image_pooled)
        if not self.use_text:
            text_seq = torch.zeros_like(text_seq)
            text_pooled = torch.zeros_like(text_pooled)

        text_padding_mask = attn_mask == 0  # [B,77]

        if self.use_coattention:
            multimodal = self.co_attention(
                image_seq, text_seq, a_padding_mask=None, b_padding_mask=text_padding_mask,
            )
        else:
            img_pool = image_seq.mean(dim=1)
            txt_pool = masked_mean(text_seq, text_padding_mask)
            multimodal = self.simple_fusion(torch.cat([img_pool, txt_pool], dim=-1))

        multimodal = self.dropout(self.fusion_norm(multimodal))

        if self.use_knowledge:
            ko = self.knowledge_retriever(
                text_query_features=text_pooled, image_query_features=image_pooled,
            )
            kf = ko["knowledge_features"]
            enhanced = self.knowledge_fusion(torch.cat([multimodal, kf], dim=-1))
            final = self.knowledge_fusion_norm(multimodal + enhanced)
        else:
            final = multimodal

        if self.use_prompt:
            class_features = self.prompt_learner()  # [2,768]
            logits = torch.matmul(
                F.normalize(final, dim=-1), F.normalize(class_features, dim=-1).t()
            ) / self.classifier_temperature
        else:
            logits = self.classifier(final)

        out = {"logits": logits, "probs": torch.softmax(logits, dim=-1)}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
