import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.models import resnet34, ResNet34_Weights
import torch.optim as optim


# load dataset to train
class ShadowDataset(Dataset):
    def __init__(self, shadow_dir, mask_dir):
        self.shadow_dir = shadow_dir
        self.mask_dir = mask_dir
        self.shadow_images = sorted(os.listdir(shadow_dir))
        self.mask_images = sorted(os.listdir(mask_dir))
        assert len(self.shadow_images) == len(self.mask_images)

    def __len__(self):
        return len(self.shadow_images)

    def __getitem__(self, idx):
        shadow_path = os.path.join(self.shadow_dir, self.shadow_images[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_images[idx])
        shadow_img = Image.open(shadow_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        shadow_tensor = TF.to_tensor(shadow_img)
        mask_tensor = TF.to_tensor(mask_img)

        mask_tensor = (mask_tensor > 0.5).float()
        return shadow_tensor, mask_tensor


# the architecture of the shadow detection model
# the UNet and ResNet are combined
class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.base_layers = list(base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        d4 = self.up4(x4)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)
        out = self.out_conv(d1)
        out = nn.functional.interpolate(
            out, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return torch.sigmoid(out)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataset = ShadowDataset("ISTD_Dataset/train/shadow", "ISTD_Dataset/train/mask")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    model = ResNetUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 16
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "ResNetUNet_shadow.pth")
    print("Model saved to ResNetUNet_shadow.pth")


if __name__ == "__main__":
    train()
