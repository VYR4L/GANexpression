from torch import nn
from datasets.fashion_mnist import get_datasets_and_loaders
from utils.plot_image import plot_generated_images
from models.generator import Generator
from models.discriminator import Discriminator
from torch.optim import Adam
import torch
from pathlib import Path


def train(epochs, batch_size, lr, beta_1, beta_2, latent_dim, save_path, examples, dim, figsize):
    ROOT_DIR = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = Adam(generator.parameters(), lr, betas=(beta_1, beta_2))
    optimizer_D = Adam(discriminator.parameters(), lr, betas=(beta_1, beta_2))

    adversarial_loss = nn.BCELoss()

    train_loader, _ = get_datasets_and_loaders(batch_size)


    # Cria o diretório para salvar as imagens geradas, se não existir
    save_path = ROOT_DIR / save_path
    save_path.mkdir(parents=True, exist_ok=True)


    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(train_loader):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)

            # Labels reais e falsas (equivalente a valid e fake no keras)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # --- Treino do Discriminador ---
            optimizer_D.zero_grad()

            # Discriminador com imagens reais
            real_preds = discriminator(imgs)
            real_loss = adversarial_loss(real_preds, valid)

            # Imagens falsas geradas
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)

            fake_preds = discriminator(gen_imgs.detach())
            fake_loss = adversarial_loss(fake_preds, fake)

            # Loss total do discriminador
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # --- Treino do Gerador (combined model no keras) ---
            optimizer_G.zero_grad()

            # Gerador quer enganar o discriminador (labels válidos)
            gen_preds = discriminator(gen_imgs)
            g_loss = adversarial_loss(gen_preds, valid)

            g_loss.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                
        # Salva imagens geradas a cada época
        plot_generated_images(epoch, generator, latent_dim=latent_dim, save_path=save_path,
                              examples=examples, dim=dim, figsize=figsize)
