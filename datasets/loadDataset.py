from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import torch

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'

__all__ = ['get_datasets_and_loaders']


def _get_fashion_mnist_dataset(dataset_dir):
    """
    Downloads and prepares the FashionMNIST dataset.

    params:
        dataset_dir (Path): Directory where the dataset will be stored.
    returns:
        Tuple[Dataset, Dataset]: Training and test datasets.
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


def _get_dataloader(dataset, batch_size=64, shuffle=True):
    """
    Creates a DataLoader for the given dataset.

    params:
        dataset (Dataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
    returns:
        DataLoader: DataLoader for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _load_dataset_from_npz(npz_dir: Path):
    """
    Loads dataset from .npz files.

    params:
        npz_dir (Path): Directory containing the .npz files.
    returns:
        Tuple[TensorDataset, TensorDataset]: Training and test datasets.
    raises:
        FileNotFoundError: If the test/validation files are not found in the specified directory.
    """
    train_data = np.load(npz_dir / 'train.npz')
    X_train = torch.tensor(train_data['X'], dtype=torch.float32).unsqueeze(1) / 255.0
    Y_train = torch.tensor(train_data['Y'], dtype=torch.long)

    train_dataset = TensorDataset(X_train, Y_train)

    try:
        test_data = np.load(npz_dir / 'test.npz')
        X_test = torch.tensor(test_data['X'], dtype=torch.float32).unsqueeze(1) / 255.0
        Y_test = torch.tensor(test_data['Y'], dtype=torch.long)
        test_dataset = TensorDataset(X_test, Y_test)
    except FileNotFoundError:
        test_dataset = None
        print(f"Test dataset not found in {npz_dir}. Using only training data.")

    return train_dataset, test_dataset


def get_datasets_and_loaders(dataset, batch_size=64):
    """
    Retrieves datasets and their corresponding DataLoaders.

    params:
        dataset (str): Name of the dataset to load (e.g., "FashionMNIST").
        batch_size (int): Number of samples per batch.
    returns:
        Tuple[DataLoader, DataLoader]: Training and test DataLoaders.
    """
    if dataset == "FashionMNIST":
        train_dataset, test_dataset = _get_fashion_mnist_dataset(DATA_DIR)
    else:
        npz_dir = DATA_DIR / dataset / "normalized"
        train_dataset, test_dataset = _load_dataset_from_npz(npz_dir)

    train_loader = _get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    if test_dataset is not None:
        test_loader = _get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    return train_loader, test_loader
