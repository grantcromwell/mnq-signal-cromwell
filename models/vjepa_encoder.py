"""
VL-JEPA Encoder for MNQ Trading Signal System

Vision-Language Joint-Embedding Predictive Architecture that combines:
- Vision Encoder: ViT processing GAF images of price action
- Text Encoder: BERT processing correlation ring descriptions
- Predictor Network: Joint embedding fusion
- Classification Head: Binary long/flat signal output

Supports:
- Self-supervised JEPA pre-training
- Supervised fine-tuning (SFT) on labeled trading signals
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VLJEPAConfig:
    """VL-JEPA model configuration"""

    vision_embed_dim: int = 768
    vision_num_heads: int = 12
    vision_num_layers: int = 12
    vision_mlp_dim: int = 3072
    vision_dropout: float = 0.1

    text_embed_dim: int = 768
    text_num_layers: int = 6
    text_num_heads: int = 8
    text_hidden_dim: int = 1024
    text_dropout: float = 0.1

    predictor_input_dim: int = 1536
    predictor_hidden_dim: int = 1024
    predictor_output_dim: int = 128
    predictor_num_layers: int = 2

    classifier_input_dim: int = 128
    classifier_hidden_dim: int = 64
    classifier_num_classes: int = 2

    use_pretrained_vision: bool = True
    use_pretrained_text: bool = True


class VisionEncoder(nn.Module):
    """
    Vision Transformer encoder for GAF images.

    Processes 3-channel GAF images (64x64) through ViT
    and produces patch embeddings for downstream processing.
    """

    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        self.config = config

        try:
            from transformers import ViTModel, ViTConfig

            vit_config = ViTConfig(
                image_size=64,
                patch_size=16,
                num_channels=3,
                hidden_size=config.vision_embed_dim,
                num_heads=config.vision_num_heads,
                intermediate_size=config.vision_mlp_dim,
                num_hidden_layers=config.vision_num_layers,
                hidden_dropout_prob=config.vision_dropout,
                attention_probs_dropout_prob=config.vision_dropout,
            )

            if config.use_pretrained_vision:
                self.model = ViTModel.from_pretrained(
                    "google/vit-base-patch16-224-in21k",
                    config=vit_config,
                    ignore_mismatched_sizes=True,
                )
            else:
                self.model = ViTModel(vit_config)

        except ImportError:
            logger.warning("Transformers not installed, using custom ViT")
            self.model = self._build_custom_vit(config)

        self.projection = nn.Linear(config.vision_embed_dim, config.vision_embed_dim)
        self.layer_norm = nn.LayerNorm(config.vision_embed_dim)
        self.dropout = nn.Dropout(config.vision_dropout)

    def _build_custom_vit(self, config: VLJEPAConfig) -> nn.Module:
        """Build custom ViT if transformers not available"""
        return nn.Identity()

    def forward(
        self, images: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for vision encoder.

        Args:
            images: Input images (B, C, H, W)
            mask: Optional attention mask

        Returns:
            Tuple of (patch_embeddings, [CLS] token embedding)
        """
        outputs = self.model(pixel_values=images)

        patch_embeddings = outputs.last_hidden_state
        cls_embedding = (
            outputs.pooler_output
            if outputs.pooler_output is not None
            else patch_embeddings[:, 0]
        )

        cls_embedding = self.layer_norm(cls_embedding)
        cls_embedding = self.projection(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)

        return patch_embeddings, cls_embedding


class TextEncoder(nn.Module):
    """
    BERT-based text encoder for correlation ring descriptions.

    Processes natural language descriptions of market correlations
    and produces text embeddings for fusion with vision features.
    """

    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        self.config = config

        try:
            from transformers import BertModel, BertConfig

            bert_config = BertConfig(
                hidden_size=config.text_embed_dim,
                num_hidden_layers=config.text_num_layers,
                num_attention_heads=config.text_num_heads,
                intermediate_size=config.text_hidden_dim,
                hidden_dropout_prob=config.text_dropout,
            )

            if config.use_pretrained_text:
                self.model = BertModel.from_pretrained(
                    "bert-base-uncased",
                    config=bert_config,
                    ignore_mismatched_sizes=True,
                )
            else:
                self.model = BertModel(bert_config)

        except ImportError:
            logger.warning("Transformers not installed, using fallback")
            self.model = nn.Identity()

        self.projection = nn.Linear(config.text_embed_dim, config.text_embed_dim)
        self.layer_norm = nn.LayerNorm(config.text_embed_dim)
        self.dropout = nn.Dropout(config.text_dropout)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for text encoder.

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: Attention mask (B, L)

        Returns:
            Tuple of (token_embeddings, [CLS] token embedding)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        token_embeddings = outputs.last_hidden_state
        cls_embedding = (
            outputs.pooler_output
            if outputs.pooler_output is not None
            else token_embeddings[:, 0]
        )

        cls_embedding = self.layer_norm(cls_embedding)
        cls_embedding = self.projection(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)

        return token_embeddings, cls_embedding


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing vision and text representations.

    Allows vision features to attend to text features and vice versa,
    enabling rich interaction between the two modalities.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention forward pass.

        Args:
            vision_features: Vision embeddings (B, N_v, D)
            text_features: Text embeddings (B, N_t, D)
            vision_mask: Optional vision mask
            text_mask: Optional text mask

        Returns:
            Tuple of (attended_vision, attended_text)
        """
        batch_size = vision_features.size(0)

        q = self.q_proj(vision_features)
        k = self.k_proj(text_features)
        v = self.v_proj(text_features)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if text_mask is not None:
            text_mask = text_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(text_mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attended_vision = torch.matmul(attn, v)
        attended_vision = attended_vision.transpose(1, 2).contiguous()
        attended_vision = attended_vision.view(batch_size, -1, self.embed_dim)
        attended_vision = self.out_proj(attended_vision)

        attended_vision = self.layer_norm(vision_features + attended_vision)

        return attended_vision, text_features


class PredictorNetwork(nn.Module):
    """
    Predictor network for joint embedding fusion.

    Takes concatenated vision and text embeddings and produces
    a unified latent representation for downstream tasks.
    """

    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        self.config = config

        layers = []
        input_dim = config.predictor_input_dim

        for _ in range(config.predictor_num_layers):
            layers.extend(
                [
                    nn.Linear(input_dim, config.predictor_hidden_dim),
                    nn.LayerNorm(config.predictor_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.vision_dropout),
                ]
            )
            input_dim = config.predictor_hidden_dim

        layers.append(nn.Linear(input_dim, config.predictor_output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through predictor network"""
        return self.network(x)


class JEPAPredictorTarget(nn.Module):
    """
    Target network for JEPA self-supervised learning.

    Uses exponential moving average (EMA) of the predictor weights
    to provide stable target representations.
    """

    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        self.config = config
        self.predictor = PredictorNetwork(config)
        self.momentum = 0.996

    @torch.no_grad()
    def update(self, source: PredictorNetwork):
        """Update target network using EMA"""
        for target_param, source_param in zip(
            self.predictor.parameters(), source.parameters()
        ):
            target_param.data = (
                self.momentum * target_param.data
                + (1 - self.momentum) * source_param.data
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (no gradient)"""
        with torch.no_grad():
            return self.predictor(x)


class BinaryClassifierHead(nn.Module):
    """
    Binary classification head for trading signal output.

    Takes joint embedding and produces:
    - Binary prediction (LONG / FLAT)
    - Confidence score (0.0 - 1.0)
    """

    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(config.classifier_input_dim, config.classifier_hidden_dim),
            nn.LayerNorm(config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.classifier_hidden_dim, config.classifier_hidden_dim // 2),
            nn.LayerNorm(config.classifier_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.classifier_hidden_dim // 2, config.classifier_num_classes),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self, joint_embedding: torch.Tensor, return_confidence: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.

        Args:
            joint_embedding: Unified representation from predictor
            return_confidence: Whether to return confidence score

        Returns:
            Dict containing logits and optionally confidence
        """
        logits = self.network(joint_embedding)

        output = {"logits": logits}

        if return_confidence:
            probs = F.softmax(logits, dim=-1)
            confidence = (
                probs[:, 1] if logits.size(1) > 1 else self.sigmoid(logits).squeeze(-1)
            )
            output["confidence"] = confidence

        return output


class VLJEPAEncoder(nn.Module):
    """
    Complete VL-JEPA Encoder combining vision and text modalities.

    Architecture:
    1. Vision Encoder: ViT on GAF images (3-day + week)
    2. Text Encoder: BERT on correlation descriptions
    3. Cross-Modal Attention: Fuse vision/text features
    4. Predictor Network: Joint embedding to latent space
    5. Classifier Head: Binary trading signal output

    Supports:
    - JEPA pre-training (self-supervised)
    - SFT fine-tuning (supervised)
    - Inference mode (signal generation)
    """

    def __init__(
        self,
        config: Optional[VLJEPAConfig] = None,
        config_path: str = "../config/config.yaml",
    ):
        super().__init__()

        if config is None:
            with open(config_path, "r") as f:
                model_config = yaml.safe_load(f)["model"]
            config = VLJEPAConfig(
                vision_embed_dim=model_config["vision"]["embed_dim"],
                vision_num_heads=model_config["vision"]["num_heads"],
                vision_num_layers=model_config["vision"]["num_layers"],
                vision_mlp_dim=model_config["vision"]["mlp_dim"],
                text_embed_dim=model_config["text"]["embed_dim"],
                text_hidden_dim=model_config["text"]["hidden_dim"],
                predictor_input_dim=model_config["vision"]["embed_dim"]
                + model_config["text"]["embed_dim"],
                predictor_hidden_dim=model_config["predictor"]["hidden_dim"],
                predictor_output_dim=model_config["predictor"]["output_dim"],
                classifier_input_dim=model_config["predictor"]["output_dim"],
                classifier_hidden_dim=model_config["classifier"]["hidden_dim"],
            )

        self.config = config
        self.vision_encoder_3day = VisionEncoder(config)
        self.vision_encoder_week = VisionEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.cross_attention = CrossModalAttention(
            config.vision_embed_dim, num_heads=8, dropout=config.vision_dropout
        )
        self.predictor = PredictorNetwork(config)
        self.classifier = BinaryClassifierHead(config)

        self.jepa_target = JEPAPredictorTarget(config)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_vision(
        self, images_3day: torch.Tensor, images_week: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode vision inputs (GAF images).

        Args:
            images_3day: 3-day GAF images (B, 3, 64, 64)
            images_week: Week GAF images (B, 3, 64, 64)

        Returns:
            Concatenated vision embeddings
        """
        _, emb_3day = self.vision_encoder_3day(images_3day)
        _, emb_week = self.vision_encoder_week(images_week)

        vision_embedding = torch.cat([emb_3day, emb_week], dim=-1)

        return vision_embedding

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode text inputs (correlation descriptions).

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: Attention mask (B, L)

        Returns:
            Text embeddings
        """
        _, text_embedding = self.text_encoder(input_ids, attention_mask)
        return text_embedding

    def forward_jepa(
        self,
        images_3day: torch.Tensor,
        images_week: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_images_3day: Optional[torch.Tensor] = None,
        target_images_week: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        JEPA forward pass for self-supervised learning.

        Args:
            images_3day: Source 3-day GAF images
            images_week: Source week GAF images
            input_ids: Text token IDs
            attention_mask: Text attention mask
            target_images_3day: Target masked images (optional)
            target_images_week: Target masked week images (optional)
            mask_ratio: Ratio of patches to mask

        Returns:
            Dict containing loss and representations
        """
        vision_embedding = self.encode_vision(images_3day, images_week)
        text_embedding = self.encode_text(input_ids, attention_mask)

        joint_embedding = torch.cat([vision_embedding, text_embedding], dim=-1)
        latent = self.predictor(joint_embedding)

        if target_images_3day is not None and target_images_week is not None:
            target_vision_embedding = self.jepa_target.predictor(
                self.encode_vision(target_images_3day, target_images_week)
            )

            jepa_loss = F.mse_loss(latent, target_vision_embedding.detach())
        else:
            jepa_loss = torch.tensor(0.0, device=latent.device)

        return {
            "latent": latent,
            "jepa_loss": jepa_loss,
            "vision_embedding": vision_embedding,
            "text_embedding": text_embedding,
        }

    def forward_sft(
        self,
        images_3day: torch.Tensor,
        images_week: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Supervised fine-tuning forward pass.

        Args:
            images_3day: 3-day GAF images (B, 3, 64, 64)
            images_week: Week GAF images (B, 3, 64, 64)
            input_ids: Text token IDs (B, L)
            attention_mask: Text attention mask (B, L)
            return_features: Whether to return intermediate features

        Returns:
            Dict containing classification output and optionally features
        """
        vision_embedding = self.encode_vision(images_3day, images_week)
        text_embedding = self.encode_text(input_ids, attention_mask)

        attended_vision, _ = self.cross_attention(
            vision_embedding.unsqueeze(1), text_embedding.unsqueeze(1)
        )

        joint_embedding = torch.cat(
            [attended_vision.squeeze(1), text_embedding], dim=-1
        )

        latent = self.predictor(joint_embedding)

        output = self.classifier(latent)

        if return_features:
            output["vision_embedding"] = vision_embedding
            output["text_embedding"] = text_embedding
            output["latent"] = latent

        return output

    def forward_inference(
        self,
        images_3day: torch.Tensor,
        images_week: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        confidence_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Inference forward pass for signal generation.

        Args:
            images_3day: 3-day GAF images
            images_week: Week GAF images
            input_ids: Text token IDs
            attention_mask: Text attention mask
            confidence_threshold: Threshold for LONG signal

        Returns:
            Dict containing signal decision and metadata
        """
        output = self.forward_sft(
            images_3day, images_week, input_ids, attention_mask, return_features=True
        )

        logits = output["logits"]
        confidence = output["confidence"].item()

        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()

        if confidence >= confidence_threshold:
            signal = "LONG"
        elif confidence >= 0.5:
            signal = "MONITOR"
        else:
            signal = "FLAT"

        return {
            "signal": signal,
            "confidence": confidence,
            "logits": logits.detach().cpu().numpy().tolist(),
            "probabilities": probs.detach().cpu().numpy().tolist(),
            "pred_class": pred_class,
        }

    @torch.no_grad()
    def update_jepa_target(self):
        """Update JEPA target network"""
        self.jepa_target.update(self.predictor)


def create_model(
    config_path: str = "../config/config.yaml",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[VLJEPAEncoder, torch.nn.Module]:
    """
    Create VL-JEPA model with optional checkpoint loading.

    Args:
        config_path: Path to config file
        checkpoint_path: Optional path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, optimizer)
    """
    model = VLJEPAEncoder(config_path=config_path).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return model, optimizer


if __name__ == "__main__":
    model, optimizer = create_model()

    model.eval()

    batch_size = 4

    images_3day = torch.randn(batch_size, 3, 64, 64)
    images_week = torch.randn(batch_size, 3, 64, 64)
    input_ids = torch.randint(0, 3000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)

    with torch.no_grad():
        output = model.forward_inference(
            images_3day, images_week, input_ids, attention_mask
        )

    print(f"Signal: {output['signal']}")
    print(f"Confidence: {output['confidence']:.4f}")
