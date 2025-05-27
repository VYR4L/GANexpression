from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'


def get_fashion_mnist_dataset(dataset_dir):
    """
    Downloads the Fashion-MNIST dataset and returns the training and test datasets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root=dataset_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=dataset_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def get_dataloader(dataset, batch_size=64, shuffle=True):
    """
    Returns a DataLoader for the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def get_datasets_and_loaders(batch_size=64):
    """
    Downloads the Fashion-MNIST dataset and returns the training and test DataLoaders.
    """
    train_dataset, test_dataset = get_fashion_mnist_dataset(DATA_DIR)

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
