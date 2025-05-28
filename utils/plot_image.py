import matplotlib.pyplot as plt
import torch
from pathlib import Path


def plot_generated_images(epoch, generator, latent_dim, examples, dim, figsize, save_path):
    generator.eval()
    device = next(generator.parameters()).device
    z = torch.randn(examples, latent_dim, device=device)

    with torch.no_grad():
        gen_imgs = generator(z).cpu()

    gen_imgs = gen_imgs.view(examples, 28, 28).numpy()

    plt.figure(figsize=figsize)
    for i in range(gen_imgs.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(gen_imgs[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    plt.tight_layout()

    filename = f"generated_images_epoch_{epoch}.png"
    filepath = Path(save_path) / filename
    plt.savefig(filepath)
    plt.close()
