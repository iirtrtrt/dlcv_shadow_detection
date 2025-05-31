import os
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms.functional as TF
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn


# the architecture of the shadow detection model
# the UNet and ResNet are combined
class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # conv, bn, relu
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # maxpool, layer1
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

    @staticmethod
    def crop_to_fit(src, target):
        _, _, h_src, w_src = src.shape
        _, _, h_tgt, w_tgt = target.shape
        crop_h = (h_src - h_tgt) // 2
        crop_w = (w_src - w_tgt) // 2
        return src[:, :, crop_h : (crop_h + h_tgt), crop_w : (crop_w + w_tgt)]

    def forward(self, x):
        x0 = self.layer0(x)  # [B,64,H/2,W/2]
        x1 = self.layer1(x0)  # [B,64,H/4,W/4]
        x2 = self.layer2(x1)  # [B,128,H/8,W/8]
        x3 = self.layer3(x2)  # [B,256,H/16,W/16]
        x4 = self.layer4(x3)  # [B,512,H/32,W/32]

        d4 = self.up4(x4)
        x3_cropped = self.crop_to_fit(x3, d4)
        d4 = torch.cat([d4, x3_cropped], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        x2_cropped = self.crop_to_fit(x2, d3)
        d3 = torch.cat([d3, x2_cropped], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        x1_cropped = self.crop_to_fit(x1, d2)
        d2 = torch.cat([d2, x1_cropped], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        x0_cropped = self.crop_to_fit(x0, d1)
        d1 = torch.cat([d1, x0_cropped], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        out = nn.functional.interpolate(
            out, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return torch.sigmoid(out)


def test(test_dir, output_dir, model_path="ResNetUNet_shadow_16.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ResNetUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    with torch.no_grad():
        for img_name in sorted(os.listdir(test_dir)):
            ext = os.path.splitext(img_name)[1].lower()
            if ext not in valid_ext:
                print(f"Skipping non-image file: {img_name}")
                continue
            img_path = os.path.join(test_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Cannot load {img_name}: {e}")
                continue

            img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

            try:
                pred_mask = model(img_tensor)[0, 0]
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue

            binary_mask = (pred_mask > 0.5).float()
            mask_img = TF.to_pil_image(binary_mask.cpu())

            mask_save_path = os.path.join(
                output_dir, f"{os.path.splitext(img_name)[0]}_mask.png"
            )
            mask_img.save(mask_save_path)
            print(f"Saved mask for: {img_name}")


if __name__ == "__main__":
    test_dir = "ISTD_test"
    output_dir = "outputs"
    test(test_dir, output_dir)
