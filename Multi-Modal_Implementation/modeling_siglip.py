from typing import Optional, Tuple
import torch
import torch.nn as nn

# Crearting congif clas bcoz model comes with different variant
class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size=768,                  #this is size of embd of transformer
            intermediate_size=3072,           # size of MLP/FF layer
            num_hidden_layers=12,             # transformer blocks
            num_attention_heads=12,           #MSA
            num_channels=3,
            image_size=224,
            patch_size=16,                    # img patch size
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int=None,      # number of img embd, or we can compare with seq len
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


#SiglipVision Model
class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [batch_size, C, H, W] ----> [batch_size, no of patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)