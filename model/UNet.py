
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.wrappers import Upsample


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, interp:bool = False):
        """Standard vanilla UNet architecture

        Args:
            in_channels (int): the number of channels in input images
            out_channels (int): the number of output classes for each pixel
        """
        super(UNet, self).__init__()

        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        self.encoder5 = ConvBlock(512, 1024)

        self.decoder4 = ConvBlock(512 * 2, 512)
        self.decoder3 = ConvBlock(256 * 2, 256)
        self.decoder2 = ConvBlock(128 * 2, 128)
        self.decoder1 = ConvBlock(64 * 2, 64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if interp: # TODO bug existed
            self.upconv4 = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv3 = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv2 = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv1 = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def adjust_size(self, target, source):
        """
            Adjust the size of the source to match the target size by either cropping or padding.
        """
        target_height, target_width = target.size(2), target.size(3)
        tensor_height, tensor_width = source.size(2), source.size(3)

        # Calculate the differences in height and width
        diffY = tensor_height - target_height
        diffX = tensor_width - target_width

        # If the source is larger in height, crop it
        if diffY > 0:
            source = self.center_crop(source, target)  # Crop the height if source is larger in height
        elif diffY < 0:  # Apply padding when source is smaller in height
            pad_height = abs(diffY)
            # Apply padding symmetrically
            source = F.pad(source, [0, 0, pad_height // 2, pad_height - pad_height // 2])  # Padding only on height

        # If the source is larger in width, crop it
        if diffX > 0:
            source = self.center_crop(source, target)  # Crop the width if source is larger in width
        elif diffX < 0:  # Apply padding when source is smaller in width
            pad_width = abs(diffX)
            # Apply padding symmetrically
            source = F.pad(source, [pad_width // 2, pad_width - pad_width // 2, 0, 0])  # Padding only on width

        return source

    def center_crop(self, source, target):
        """
            Crop the source from the center to match the target size.
        """
        target_height, target_width = target.size(2), target.size(3)
        tensor_height, tensor_width = source.size(2), source.size(3)

        # Calculate the difference
        diffY = tensor_height - target_height
        diffX = tensor_width - target_width

        # Calculate the crop indices for both height and width
        cropY_start = diffY // 2
        cropY_end = tensor_height - diffY // 2
        cropX_start = diffX // 2
        cropX_end = tensor_width - diffX // 2

        return source[:, :, cropY_start:cropY_end, cropX_start:cropX_end]

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        # Decoder
        dec4 = self.upconv4(enc5)
        dec4 = self.adjust_size(enc4, dec4)  # Adjust size before concatenating
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.adjust_size(enc3, dec3)  # Adjust size before concatenating
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.adjust_size(enc2, dec2)  # Adjust size before concatenating
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.adjust_size(enc1, dec1)  # Adjust size before concatenating
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection
        dec1 = self.decoder1(dec1)

        # Final convolution to get the output
        out = self.final_conv(dec1)
        return out
