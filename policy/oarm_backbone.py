import torch
import torch.nn as nn

from OARM.utils.yopo_compat import ensure_yopo_path


class OARMBackbone(nn.Module):
    """Lightweight depth encoder that preserves YOPO's V x H lattice output.

    YOPO's original ResNet head maps a 96x160 depth image to a 3x5 lattice.
    This local backbone avoids touching/copying the baseline model code while
    producing the same spatial contract for OARM experiments.
    """

    def __init__(self, output_dim: int, vertical_num: int, horizon_num: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((vertical_num, horizon_num)),
            nn.Conv2d(128, output_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.encoder(depth)


class YOPOOriginalBackbone(nn.Module):
    """Adapter around YOPO's original ResNet18 depth backbone.

    This keeps the OARM head/loss unchanged while isolating whether gains come
    from reaction-margin supervision or from changing the visual encoder.
    """

    def __init__(self, output_dim: int):
        super().__init__()
        ensure_yopo_path()
        from policy.models.backbone import YopoBackbone

        self.encoder = YopoBackbone(output_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.encoder(depth)
