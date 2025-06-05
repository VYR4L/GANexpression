from pathlib import Path
from PIL import Image
import numpy as np


def save_dataset_as_npz(dataset_dir: Path, image_size=(48, 48)):
    """
    Saves the dataset in .npz format after normalizing images.

    params:
        dataset_dir (Path): Directory containing the dataset.
        image_size (tuple): Size to which images will be resized.
    """

    normalized_dir = dataset_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    existing_files = {f.name for f in normalized_dir.glob("*.npz")}
    expected_files = {"train.npz", "validation.npz", "test.npz"}

    if existing_files >= expected_files:
        print(f"Dataset already normalized Skipping.")
        return

    for split in ["train", "validation", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        X, Y = [], []
        class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        labeml_map = {class_name: i for i, class_name in enumerate(class_names)}

        for class_name in class_names:
            for img_path in (split_dir / class_name).glob("*.png"):
                img = Image.open(img_path).convert("L")
                img = img.resize(image_size)
                X.append(np.array(img))
                Y.append(labeml_map[class_name])

        np.savez_compressed(
            normalized_dir / f"{split}.npz",
            X=np.array(X),
            Y=np.array(Y)
        )
        print(f"Saved {split} dataset to {normalized_dir / f'{split}.npz'}")

