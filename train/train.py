from torch import nn
from datasets.loadDataset import get_datasets_and_loaders
from utils.plotImage import plot_generated_images
from models.generator import Generator
from models.discriminator import Discriminator
from torch.optim import Adam
from utils.dataSetNormalizator import save_dataset_as_npz
import torch
from pathlib import Path


def train(dataset, epochs, batch_size, lr, beta_1, beta_2, latent_dim, save_path, examples, dim, figsize):
    """
    Train a Generative Adversarial Network (GAN) on the specified dataset.

    params:
        dataset (str): Name of the dataset to use for training.
        epochs (int): Number of epochs to train the GAN.
        batch_size (int): Size of each batch during training.
        lr (float): Learning rate for the optimizer.
        beta_1 (float): Beta 1 parameter for the Adam optimizer.
        beta_2 (float): Beta 2 parameter for the Adam optimizer.
        latent_dim (int): Dimensionality of the latent space for the generator.
        save_path (str): Directory to save generated images.
        examples (int): Number of images to generate per epoch.
        dim (tuple): Dimensions of the generated image grid.
        figsize (tuple): Size of the figure for displaying generated images.
    """
    ROOT_DIR = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dataset_as_npz(dataset)
    train_loader, _ = get_datasets_and_loaders(dataset, batch_size)

    num_classes = len(set(label.item() for _, label in train_loader.dataset))

    generator = Generator(num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)

    optimizer_G = Adam(generator.parameters(), lr, betas=(beta_1, beta_2))
    optimizer_D = Adam(discriminator.parameters(), lr, betas=(beta_1, beta_2))

    adversarial_loss = nn.BCELoss()

    save_path = ROOT_DIR / save_path
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            real_labels = labels.to(device)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            optimizer_D.zero_grad()

            real_preds = discriminator(real_imgs, real_labels)
            real_loss = adversarial_loss(real_preds, valid)

            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)

            fake_preds = discriminator(gen_imgs.detach(), gen_labels)
            fake_loss = adversarial_loss(fake_preds, fake)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            gen_preds = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(gen_preds, valid)

            g_loss.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        fixed_z = torch.randn(examples, latent_dim, device=device)
        fixed_labels = torch.arange(0, examples, device=device) % num_classes
        plot_generated_images(epoch, generator, latent_dim=latent_dim, save_path=save_path,
                              examples=examples, dim=dim, figsize=figsize,
                              num_classes=num_classes, z=fixed_z, labels=fixed_labels)

    torch.save(generator.state_dict(), save_path / "generator.pth")
    torch.save(discriminator.state_dict(), save_path / "discriminator.pth")
