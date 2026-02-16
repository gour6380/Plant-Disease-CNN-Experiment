import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
	A simple baseline CNN classifier with two conv blocks and an MLP head.

	Architecture:
	- Feature extractor: (Conv2d -> ReLU -> BatchNorm -> MaxPool) x2
	- Classifier: Flatten -> Linear(262144->128) -> ReLU -> Dropout ->
	  Linear(128->64) -> ReLU -> Dropout -> Linear(64->38)

	Args:
		input_features: Number of input channels (e.g., 3 for RGB).

	Inputs:
		x: Tensor of shape (N, C, H, W).

	Returns:
		Logits tensor of shape (N, 38).
	"""
    def __init__(self, input_features) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_features, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=262144, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(in_features=64, out_features=38),

        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualGNBlock(nn.Module):
    """
	A residual convolutional block using GroupNorm and optional projection shortcut.

	Applies two 3x3 convolutions with GroupNorm and ReLU, then adds a shortcut
	connection. If `stride != 1` or channel dimensions differ, uses a 1x1
	projection (Conv + GroupNorm) for the shortcut.

	Args:
		in_ch: Number of input channels.
		out_ch: Number of output channels.
		stride: Stride for the first convolution and shortcut projection.

	Inputs:
		x: Input tensor of shape (N, in_ch, H, W).

	Returns:
		Output tensor of shape (N, out_ch, H/stride, W/stride).
	"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class AdvancedCNN(nn.Module):
    """
	A deeper CNN classifier with residual GroupNorm blocks and global pooling head.

	Architecture:
	- Stem: 3x3 Conv -> GroupNorm -> ReLU
	- Stages 1-4: ResidualGNBlock stacks with downsampling at stage2-4 (stride=2)
	- Head: AdaptiveAvgPool -> Flatten -> Dropout -> Linear(256 -> num_classes)

	Args:
		num_classes: Number of output classes.
		in_channels: Number of input image channels.
		dropout: Dropout probability applied before the final linear layer.

	Inputs:
		x: Input tensor of shape (N, in_channels, H, W).

	Returns:
		Logits tensor of shape (N, num_classes).
	"""
    def __init__(self, num_classes: int = 38, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResidualGNBlock(32, 32, stride=1),
            ResidualGNBlock(32, 32, stride=1),
        )  # 256x256

        self.stage2 = nn.Sequential(
            ResidualGNBlock(32, 64, stride=2),
            ResidualGNBlock(64, 64, stride=1),
        )  # 128x128

        self.stage3 = nn.Sequential(
            ResidualGNBlock(64, 128, stride=2),
            ResidualGNBlock(128, 128, stride=1),
        )  # 64x64

        self.stage4 = nn.Sequential(
            ResidualGNBlock(128, 256, stride=2),
            ResidualGNBlock(256, 256, stride=1),
        )  # 32x32

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)
