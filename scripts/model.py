"""
model.py — Foundation Model Architecture for Crop Mapping
==========================================================
Author: Samuel Appiah Kubi
Copernicus Master's in Digital Earth, Paris Lodron University Salzburg

Two-phase fine-tuning:
  Phase 1 → Freeze backbone, train classification head only
  Phase 2 → Unfreeze later layers, train end-to-end with very low LR

Supported backbones:
  1. Clay (EO foundation model, HuggingFace)
  2. ResNet-50 (ImageNet pre-trained, always available as fallback)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import load_config

logger = logging.getLogger("crop_mapping.model")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch not installed. Run: pip install torch")


# =============================================================================
# Classification head
# =============================================================================

if TORCH_AVAILABLE:

    class CropClassificationHead(nn.Module):
        """
        MLP classification head to replace the pre-trained head.

        Architecture:
          LayerNorm → Linear(in, 256) → GELU → Dropout → Linear(256, num_classes)
        """

        def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
            super().__init__()
            self.norm = nn.LayerNorm(in_features)
            self.fc1 = nn.Linear(in_features, 256)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(256, num_classes)

            # Initialise final layer with small weights
            nn.init.normal_(self.fc2.weight, std=0.01)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.norm(x)
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            return self.fc2(x)


    # =========================================================================
    # ResNet-50 backbone (always available fallback)
    # =========================================================================

    class ResNet50CropModel(nn.Module):
        """
        ResNet-50 (ImageNet pre-trained) fine-tuned for crop mapping.
        Adapted for multi-band input (> 3 channels) by replacing conv1.
        """

        def __init__(
            self,
            num_classes: int,
            in_channels: int = 8,
            freeze_ratio: float = 0.85,
            dropout: float = 0.3,
        ):
            super().__init__()
            from torchvision.models import resnet50, ResNet50_Weights

            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # Replace conv1 to accept in_channels (not just 3)
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False,
            )
            # Initialise new channels by averaging ImageNet weights
            with torch.no_grad():
                if in_channels <= 3:
                    new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
                else:
                    # Tile the 3-channel weights across all in_channels
                    for i in range(in_channels):
                        new_conv.weight[:, i] = old_conv.weight[:, i % 3]
            backbone.conv1 = new_conv

            # Remove original classification head
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

            # Freeze layers
            self._freeze_layers(freeze_ratio)

            # Custom head
            self.head = CropClassificationHead(feature_dim, num_classes, dropout)

            logger.info(
                f"ResNet-50 model: {num_classes} classes, "
                f"{in_channels} input channels, "
                f"freeze_ratio={freeze_ratio}"
            )

        def _freeze_layers(self, ratio: float):
            """Freeze the first `ratio` fraction of named parameters."""
            params = list(self.backbone.named_parameters())
            n_freeze = int(len(params) * ratio)
            for i, (name, param) in enumerate(params):
                param.requires_grad = i >= n_freeze
            frozen = sum(1 for _, p in self.backbone.named_parameters() if not p.requires_grad)
            total = sum(1 for _ in self.backbone.parameters())
            logger.info(f"  Frozen {frozen}/{total} backbone parameters")

        def unfreeze_top_layers(self, n_layers: int = 2):
            """
            Unfreeze the last n_layers ResNet layers for Phase 2 fine-tuning.
            ResNet-50 layers: layer1, layer2, layer3, layer4, avgpool
            """
            layer_names = [f"layer{4 - i}" for i in range(n_layers)]
            layer_names.append("avgpool")
            for name, param in self.backbone.named_parameters():
                for ln in layer_names:
                    if name.startswith(ln):
                        param.requires_grad = True
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            logger.info(f"  Phase 2: unfroze {layer_names}, trainable params: {trainable:,}")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)  # (B, 2048)
            return self.head(features)   # (B, num_classes)


    # =========================================================================
    # Clay EO Foundation Model
    # =========================================================================

    class ClayBackbone(nn.Module):
        """
        Wrapper around the Clay Earth Observation Foundation Model.

        Clay is a Vision Transformer pre-trained on Sentinel-2, Landsat,
        SAR, and DEM data covering the globe.

        Reference: made-with-clay/Clay on HuggingFace
        """

        def __init__(
            self,
            model_id: str = "made-with-clay/Clay",
            variant: str = "small",
            in_channels: int = 8,
        ):
            super().__init__()
            self.feature_dim = None
            self.model = None

            try:
                self._load_clay(model_id, variant, in_channels)
            except Exception as e:
                logger.warning(f"Clay model failed to load: {e}")
                logger.warning("Falling back to ResNet-50.")
                self.model = None

        def _load_clay(self, model_id: str, variant: str, in_channels: int):
            """
            Attempt to load Clay from HuggingFace transformers.
            Clay uses a ViT-based architecture — we extract the [CLS] token as features.
            """
            try:
                from transformers import AutoModel, AutoConfig
                config = AutoConfig.from_pretrained(model_id)
                self.model = AutoModel.from_pretrained(model_id)
                # Try to infer feature dim
                self.feature_dim = getattr(config, "hidden_size", 768)
                logger.info(f"Clay model loaded from HuggingFace: {model_id}  feature_dim={self.feature_dim}")
            except Exception as e:
                # Try clay-specific package
                try:
                    import clay  # noqa
                    self.model = clay.ClayMAEModel.from_pretrained(f"clay-{variant}")
                    self.feature_dim = self.model.embed_dim
                    logger.info(f"Clay model loaded via clay package: feature_dim={self.feature_dim}")
                except Exception as e2:
                    raise RuntimeError(f"Could not load Clay: HF error={e}, clay error={e2}")

        @property
        def is_available(self) -> bool:
            return self.model is not None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through Clay backbone.

            Parameters
            ----------
            x : torch.Tensor  shape (B, C, H, W)

            Returns
            -------
            features : torch.Tensor  shape (B, feature_dim)
            """
            if self.model is None:
                raise RuntimeError("Clay backbone not loaded")

            # Clay expects dict input: {"pixels": ..., "time": ..., "latlon": ...}
            # For simplicity, pass pixels directly if possible
            try:
                out = self.model(pixel_values=x)
                # Extract [CLS] token or mean of patch tokens
                if hasattr(out, "last_hidden_state"):
                    features = out.last_hidden_state[:, 0, :]  # CLS token
                elif hasattr(out, "pooler_output"):
                    features = out.pooler_output
                else:
                    features = out[0][:, 0, :]
                return features
            except Exception:
                # Fallback: direct call
                out = self.model(x)
                if isinstance(out, torch.Tensor):
                    if out.ndim == 3:
                        return out[:, 0, :]   # CLS token
                    return out
                return out[0][:, 0, :]


    class ClayCropModel(nn.Module):
        """
        Full Clay + classification head model.
        Falls back to ResNet-50 if Clay is unavailable.
        """

        def __init__(
            self,
            num_classes: int,
            in_channels: int = 8,
            freeze_ratio: float = 0.85,
            model_id: str = "made-with-clay/Clay",
            variant: str = "small",
            dropout: float = 0.3,
        ):
            super().__init__()
            backbone = ClayBackbone(model_id, variant, in_channels)

            if not backbone.is_available:
                logger.warning("Clay unavailable. Using ResNet-50 fallback.")
                self._fallback = ResNet50CropModel(
                    num_classes, in_channels, freeze_ratio, dropout
                )
                self._use_fallback = True
                return

            self._use_fallback = False
            self.backbone = backbone
            self._freeze_backbone(freeze_ratio)

            feature_dim = backbone.feature_dim
            self.head = CropClassificationHead(feature_dim, num_classes, dropout)
            logger.info(f"ClayCropModel ready: feature_dim={feature_dim}")

        def _freeze_backbone(self, ratio: float):
            params = list(self.backbone.model.named_parameters())
            n_freeze = int(len(params) * ratio)
            for i, (_, param) in enumerate(params):
                param.requires_grad = i >= n_freeze
            frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
            total = sum(1 for _ in self.backbone.parameters())
            logger.info(f"  Clay backbone: frozen {frozen}/{total} parameters")

        def unfreeze_for_phase2(self, n_transformer_blocks: int = 2):
            """Unfreeze the last N transformer blocks for Phase 2 fine-tuning."""
            if self._use_fallback:
                self._fallback.unfreeze_top_layers(n_transformer_blocks)
                return
            # Transformer blocks usually named 'encoder.layer' or 'blocks'
            params = list(self.backbone.model.named_parameters())
            # Unfreeze last 10% of parameters
            n_unfreeze = max(1, int(len(params) * 0.1))
            for _, param in params[-n_unfreeze:]:
                param.requires_grad = True
            logger.info(f"  Phase 2: unfroze last {n_unfreeze} backbone params")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self._use_fallback:
                return self._fallback(x)
            features = self.backbone(x)
            return self.head(features)


    # =========================================================================
    # Model factory
    # =========================================================================

    def build_model(cfg: Dict) -> nn.Module:
        """
        Build the appropriate model based on config settings.

        Returns a model with:
          - backbone frozen at cfg.foundation_model.freeze_ratio
          - classification head for cfg.classes.num_classes classes
        """
        fm_cfg = cfg["foundation_model"]
        model_name = fm_cfg.get("name", "resnet50").lower()
        num_classes = cfg["classes"]["num_classes"]
        in_channels = cfg["patches"]["bands"]
        freeze_ratio = fm_cfg.get("freeze_ratio", 0.85)

        if model_name == "clay":
            logger.info("Building Clay foundation model…")
            model = ClayCropModel(
                num_classes=num_classes,
                in_channels=in_channels,
                freeze_ratio=freeze_ratio,
                model_id=fm_cfg.get("hf_model_id", "made-with-clay/Clay"),
                variant=fm_cfg.get("variant", "small"),
            )
        else:  # resnet50 or fallback
            logger.info("Building ResNet-50 model…")
            model = ResNet50CropModel(
                num_classes=num_classes,
                in_channels=in_channels,
                freeze_ratio=freeze_ratio,
            )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model ready — total params: {total_params:,}  "
            f"trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)"
        )
        return model


    # =========================================================================
    # Parameter groups for differential learning rates
    # =========================================================================

    def get_optimizer(model: nn.Module, cfg: Dict) -> "torch.optim.Optimizer":
        """
        Build AdamW with two parameter groups:
          - Head: high learning rate
          - Backbone: low learning rate (for Phase 2)
        """
        train_cfg = cfg["training"]
        lr_head = train_cfg["learning_rate_head"]
        lr_backbone = train_cfg["learning_rate_backbone"]
        wd = train_cfg.get("weight_decay", 1e-4)

        head_params, backbone_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": head_params, "lr": lr_head, "name": "head"},
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
        logger.info(
            f"Optimizer: AdamW  lr_head={lr_head}  lr_backbone={lr_backbone}  wd={wd}"
        )
        return optimizer


    def get_scheduler(optimizer, cfg: Dict) -> "torch.optim.lr_scheduler._LRScheduler":
        """Build learning rate scheduler from config."""
        scheduler_type = cfg["training"].get("scheduler", "reduce_on_plateau")
        patience = cfg["training"].get("patience", 5)
        min_lr = cfg["training"].get("min_lr", 1e-6)

        if scheduler_type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5,
                patience=patience // 2, min_lr=min_lr, verbose=True,
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["training"]["epochs"], eta_min=min_lr,
            )
        else:
            return None


    def get_loss(cfg: Dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """
        CrossEntropy loss with optional class weighting.
        ignore_index=255 corresponds to unlabelled pixels.
        """
        if cfg["training"].get("class_weighting", True) and class_weights is not None:
            logger.info("Using weighted CrossEntropy loss")
            return nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        return nn.CrossEntropyLoss(ignore_index=255)
