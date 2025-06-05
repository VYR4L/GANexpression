import matplotlib.pyplot as plt
import torch
from pathlib import Path


def plot_generated_images(epoch, generator, latent_dim, examples, dim, figsize, save_path, num_classes, z=None, labels=None):
    """
    Plots and saves generated images from the generator model.

    params:
        epoch (int): Current epoch number for naming the saved image.
        generator (nn.Module): The generator model to use for generating images.
        latent_dim (int): Dimension of the latent space.
        examples (int): Number of images to generate.
        dim (tuple): Dimensions of the subplot grid (rows, columns).
        figsize (tuple): Size of the figure for plotting.
        save_path (str or Path): Directory where the generated images will be saved.
        num_classes (int): Total number of classes (for label sampling).
        z (torch.Tensor, optional): Predefined latent vectors. If None, random vectors will be generated.
        labels (torch.Tensor, optional): Predefined labels for the generated images. If None, random labels will be generated.
    """  
    generator.eval()

    device = next(generator.parameters()).device

    with torch.no_grad():
        if z is None:
            z = torch.randn(examples, latent_dim).to(device)
        if labels is None:
            labels = torch.randint(0, num_classes, (examples,), device=device)

        gen_imgs = generator(z, labels)
        gen_imgs = gen_imgs.cpu().squeeze(1)  # (B, H, W)

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
