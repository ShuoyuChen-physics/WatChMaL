"""
Here is a Swin Transformer model.
"""

import torch
import torch.nn as nn
import timm


class SwinRegressor(nn.Module):
    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        img_size=(192, 192),
        in_chans=2,
        num_output_channels=3,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_output_channels,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )
        self.output_dim = num_output_channels

    def forward(self, x):
        out = self.vit(x)
        return out


class MultiTaskSwin(nn.Module):
    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        img_size=(192, 192),
        in_chans=2,
        drop_path_rate=0.0,
        task_output_dims={"positions": 3, "directions": 3, "energies": 1},
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            num_classes=0,
        )
        num_features = self.backbone.num_features
        self.task_heads = nn.ModuleDict()
        for task_name, output_dim in task_output_dims.items():
            self.task_heads[task_name] = nn.Linear(num_features, output_dim)

    def forward(self, x):
        features = self.backbone(x)
        outputs = {
            task_name: head(features) for task_name, head in self.task_heads.items()
        }
        return outputs
