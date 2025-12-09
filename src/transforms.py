import torch
from torchvision import transforms
from torchvision.transforms import functional
from torchvision.transforms.functional import InterpolationMode


def get_transforms(*, train: bool = True, normalize: bool = True) -> transforms.Compose:
    """Get the transformations to apply to images based on the mode.

    Args:
        train (bool): If True, returns transformations for training images. Otherwise, for validation images.
        normalize (bool): If True, applies normalization to the images.

    Returns:
        transforms.Compose: Composed transformations for images.
    """
    img_mean = [0.64041256, 0.36125767, 0.31330117]
    img_std = [0.18983584, 0.15554344, 0.14093774]

    def flip(img: torch.Tensor) -> torch.Tensor:
        h_thres = 0.25
        v_thres = 0.5
        r = torch.rand(1).item()
        if r < h_thres:
            return functional.hflip(img)
        if r < v_thres:
            return functional.vflip(img)
        return img

    def rotate(img: torch.Tensor) -> torch.Tensor:
        angles = [0, 90, 180, 270]
        angle = angles[torch.randint(0, len(angles), (1,)).item()]
        return functional.rotate(img, angle)

    if train:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=InterpolationMode.LANCZOS),
                transforms.Lambda(flip),
                transforms.Lambda(rotate),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=InterpolationMode.LANCZOS),
                transforms.ToTensor(),
            ]
        )
    if normalize:
        transform = transforms.Compose(
            [
                transform,
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        )

    return transform


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverses the normalization applied to a tensor.

    Args:
        tensor (torch.Tensor): The normalized tensor to be denormalized.

    Returns:
        torch.Tensor: The denormalized tensor.
    """
    img_mean = [0.64041256, 0.36125767, 0.31330117]
    img_std = [0.18983584, 0.15554344, 0.14093774]

    mean = torch.tensor(img_mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(img_std, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean
