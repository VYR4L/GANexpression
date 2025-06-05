from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 48, 48), num_classes=7):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.init_size = img_shape[1] // 4  # Assuming the image is downsampled by a factor of 4
        self.layer_1 = nn.Sequential(nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2))
       
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.Relu(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.Relu(inplace=True),

            nn.conv2d(64, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, img, label):
        label_input = self.label_embedding(label)
        x = torch.cat([img, label_input], dim=1)

        out = self.layer_1(x)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img