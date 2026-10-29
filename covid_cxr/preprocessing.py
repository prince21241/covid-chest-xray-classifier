"""Image preprocessing shared by training, validation, testing, and prediction.

Every image is converted to RGB and letterboxed to 224 x 224 so the full X-ray
stays in frame. Validation and test images are not augmented. Training images
receive a small rotation and translation only.
"""

from __future__ import annotations

from PIL import Image

from covid_cxr.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PAD_FILL,
    TRAIN_ROTATION_DEGREES,
    TRAIN_TRANSLATE,
)


class ResizeWithPad:
    """Resize the long side to ``size`` and pad the short side, centered."""

    def __init__(self, size: int = IMAGE_SIZE, fill: int = PAD_FILL) -> None:
        if size < 2:
            raise ValueError("Image size must be at least 2.")
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError("Image has no pixels.")
        if width >= height:
            new_width = self.size
            new_height = max(1, int(round(height * self.size / width)))
            new_height = min(new_height, self.size)
        else:
            new_height = self.size
            new_width = max(1, int(round(width * self.size / height)))
            new_width = min(new_width, self.size)
        resized = image.resize((new_width, new_height), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        canvas.paste(resized, (left, top))
        return canvas


def preprocessing_config() -> dict:
    """Configuration stored beside the model weights."""
    return {
        "image_size": IMAGE_SIZE,
        "color_mode": "RGB",
        "resize": "letterbox_pad",
        "aspect_ratio": "preserved",
        "full_image_retained": True,
        "interpolation": "bilinear",
        "pad_fill": PAD_FILL,
        "imagenet_mean": [float(value) for value in IMAGENET_MEAN],
        "imagenet_std": [float(value) for value in IMAGENET_STD],
        "train_augmentation": {
            "rotation_degrees": TRAIN_ROTATION_DEGREES,
            "translate_fraction": TRAIN_TRANSLATE,
            "horizontal_flip": False,
            "vertical_flip": False,
            "random_resized_crop": False,
        },
        "evaluation_augmentation": None,
    }


def build_transforms(train: bool, config: dict | None = None):
    """Build the tensor transform.

    Importing torchvision happens here so data preparation and geometry tests
    can run before PyTorch is installed.
    """
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    cfg = config or preprocessing_config()
    if cfg.get("resize") != "letterbox_pad" or cfg.get("color_mode") != "RGB":
        raise ValueError("Checkpoint preprocessing is not the letterbox RGB pipeline from this project.")
    operations = [ResizeWithPad(int(cfg["image_size"]), fill=int(cfg["pad_fill"]))]
    if train:
        augmentation = cfg["train_augmentation"]
        translate = float(augmentation["translate_fraction"])
        operations.append(
            transforms.RandomAffine(
                degrees=float(augmentation["rotation_degrees"]),
                translate=(translate, translate),
                interpolation=InterpolationMode.BILINEAR,
                fill=int(cfg["pad_fill"]),
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg["imagenet_mean"], cfg["imagenet_std"]),
        ]
    )
    return transforms.Compose(operations)
