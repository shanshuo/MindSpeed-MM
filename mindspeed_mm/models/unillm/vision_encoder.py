from typing import Optional, Any, Tuple, List, Union

import torch
from torch import nn
import torchvision
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

from .siglip_vit import create_siglip_vit



class GenEncoder(nn.Module):
    def __init__(
        self,
        checkpoint: str = None,
        checkpoint_enc: str = None,
        checkpoint_dec: str = None,
        tokenizer_config: dict[str, Any] = None,
        image_token_size: int = None,
        n_embed: int = None,
        device: str = "npu",
        dtype: str = "bfloat16"
    ):
        super().__init__()
        if checkpoint_enc is not None:
            self.encoder = CausalVideoTokenizer(checkpoint_enc=checkpoint_enc, device=device, dtype=dtype)
        if checkpoint_dec is not None:
            self.decoder = CausalVideoTokenizer(checkpoint_dec=checkpoint_dec, device=device, dtype=dtype)
        self.embedding = nn.Embedding(image_token_size, n_embed)
    
    def encode(self, images):
        if images.dim() == 4:
            images = images.unsqueeze(2)
        indices, codes = self.encoder.encode(images)
        b = indices.shape[0]
        images_ids = indices.view(b, -1)
        image_embeds = self.embedding(images_ids)
        return images_ids, image_embeds, codes

    def decode(self, indices):
        reconstructed_tensor = self.decoder.decode(indices)
        return reconstructed_tensor


class UndEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: Union[Tuple[int, int], int] = 336,
        select_feature: str = "patch",
        select_layer: int = -2,
        select_layers: list = None,
        ckpt_path: str = "",
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()

        if not model_name.startswith("siglip") and select_feature != "same":
            raise NotImplementedError("Only support siglip now!!!")

        self.model_name = model_name
        self.select_feature = select_feature
        self.select_layer = select_layer
        self.select_layers = select_layers

        vision_tower_params = {
            "model_name": model_name,
            "image_size": image_size,
            "ckpt_path": ckpt_path,
            "select_layer": select_layer,
        }
        vision_tower_params.update(kwargs)

        self.vision_tower = create_siglip_vit(**vision_tower_params)

        if pixel_mean is not None and pixel_std is not None:
            image_norm = torchvision.transforms.Normalize(
                mean=pixel_mean, std=pixel_std
            )
        else:
            image_norm = None

        self.image_norm = image_norm
    
    def forward(self, images):
        """

        Args:
            images (torch.Tensor): [b, 3, H, W]

        Returns:
            image_features (torch.Tensor): [b, n_patch, d]
        """

        if self.image_norm is not None:
            images = self.image_norm(images)

        image_forward_outs = self.vision_tower(images)

        if isinstance(image_forward_outs, torch.Tensor):
            # the output has been the self.select_layer"s features
            image_features = image_forward_outs
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        return image_features
