from torchvision import transforms as t

def build_transforms(
    image_height: int,
    image_width: int,
    mean: tuple[float, ...],
    std: tuple[float, ...]) -> tuple[t.Compose, t.Compose]:
    """
	Create torchvision transforms for training and evaluation.

	Builds two `torchvision.transforms.Compose` pipelines:
	- Train: resize + basic augmentation + tensor conversion + normalization
	- Eval: resize + tensor conversion + normalization

	Args:
		image_height: Target image height after resizing.
		image_width: Target image width after resizing.
		mean: Channel-wise normalization mean.
		std: Channel-wise normalization standard deviation.

	Returns:
		(train_tfm, eval_tfm) as a tuple of composed transforms.
	"""

    train_tfm = t.Compose(
        [
            t.Resize((image_height, image_width)),
            t.RandomHorizontalFlip(p=0.5),
            t.RandomRotation(degrees=10),
            t.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            t.ToTensor(),
            t.Normalize(mean=mean, std = std),
        ]
    )

    eval_tfm = t.Compose(
        [
            t.Resize((image_height, image_width)),
            t.ToTensor(),
            t.Normalize(mean=mean, std = std),
        ]
    )

    return train_tfm, eval_tfm