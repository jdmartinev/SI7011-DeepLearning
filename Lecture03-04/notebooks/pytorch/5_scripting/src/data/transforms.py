"""Transforms matching the ResNet18 ImageNet pre-training protocol."""
from torchvision.models import ResNet18_Weights
from torchvision import transforms


def get_inference_transform() -> transforms.Compose:
    """The canonical preprocessing used during ImageNet pre-training.

    Using the same statistics at fine-tuning time is important: the
    pre-trained weights encode assumptions about input distribution.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    return weights.transforms()


def get_augmented_transform() -> transforms.Compose:
    """Adds random crop and horizontal flip on top of the ImageNet preprocessing.

    Useful for stage-2 full fine-tuning where the model can benefit from
    more variance in the training data.
    """
    base = get_inference_transform()
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            base,
        ]
    )
