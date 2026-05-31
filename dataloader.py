import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size=64):
    # Convert Image to Tensor and Normalize
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Download Data
    training_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
