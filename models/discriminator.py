from torch import nn
import torch


class Discriminator(nn.Module):
    def __init__(self, image_shape=(1, 48, 48), num_classes=7):
        super().__init__()
        self.image_shape = image_shape
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Conv2d(1 + num_classes, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img, label):
        label_input = self.label_embedding(label)
        label_input = label_input.view(label_input.size(0), self.num_classes, 1, 1)    
        label_input = label_input.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, label_input], dim=1)
        return self.model(x)