from typing import Tuple

import torch
import torchvision
from torchvision.transforms import Compose


def build_efficientnet_b0_embedder(device: torch.device) -> Tuple[torch.nn.Module, Compose, int]:
    # Pretrained EfficientNetB0 backbone, global pooled embedding
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    model.eval()
    # Remove classifier; take features before final FC
    model.classifier = torch.nn.Identity()
    model.to(device)

    # Use official weight-defined transforms to avoid metadata key issues
    preprocess = weights.transforms()

    # Embedding dim for EfficientNetB0 features
    embedding_dim = 1280
    return model, preprocess, embedding_dim


