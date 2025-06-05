import argparse
from train.train import train


def main():
    parser = argparse.ArgumentParser(description="Train a GAN on Fashion MNIST")
    parser.add_argument("--dataset", type=str, default="FashionMNIST", help="Dataset to use for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate for the optimizer")
    parser.add_argument("--beta_1", type=float, default=0.5, help="Beta 1 for Adam optimizer")
    parser.add_argument("--beta_2", type=float, default=0.999, help="Beta 2 for Adam optimizer")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
    parser.add_argument("--save_path", type=str, default="generated_images", help="Path to save generated images")
    parser.add_argument("--examples", type=int, default=100, help="Number of images to generate per epoch")
    parser.add_argument("--dim", type=int, nargs=2, default=(10, 10), help="Dimensions of the generated image grid")
    parser.add_argument("--figsize", type=float, nargs=2, default=(10, 10), help="Figure size for generated images")
    args = parser.parse_args()

    # TODO: Adaptar o GAN para ser condicional (cGAN)

    train(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        latent_dim=args.latent_dim,
        save_path=args.save_path,
        examples=args.examples,
        dim=args.dim,
        figsize=args.figsize
    )


if __name__ == "__main__":
    main()