
import torch
import torch.nn as nn
import torch.nn.functional as F

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)  # 添加dropout
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch, dropout_rate=0.0):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate)  # 添加dropout

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.dropout(x1)  # 应用dropout
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FCN(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),  # 添加dropout
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),  # 添加dropout
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),  # 添加dropout
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.fcn(x)


class UNet(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(UNet, self).__init__()
        
        self.inc = nn.Sequential(
            single_conv(2, 32, dropout_rate),
            single_conv(32, 32, dropout_rate)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(32, 64, dropout_rate),
            single_conv(64, 64, dropout_rate),
            single_conv(64, 64, dropout_rate)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(64, 128, dropout_rate),
            single_conv(128, 128, dropout_rate),
            single_conv(128, 128, dropout_rate),
            single_conv(128, 128, dropout_rate),
            single_conv(128, 128, dropout_rate),
            single_conv(128, 128, dropout_rate)
        )

        self.up1 = up(128, dropout_rate)
        self.conv3 = nn.Sequential(
            single_conv(64, 64, dropout_rate),
            single_conv(64, 64, dropout_rate),
            single_conv(64, 64, dropout_rate)
        )

        self.up2 = up(64, dropout_rate)
        self.conv4 = nn.Sequential(
            single_conv(32, 32, dropout_rate),
            single_conv(32, 32, dropout_rate)
        )

        self.outc = outconv(32, 1)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class Encoder(nn.Module):
    def __init__(self, input_channels, dropout_rate=0.0):
        super(Encoder, self).__init__()
        
        # Encoder layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
    def forward(self, x):
        # Encoder block 1
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        skip1 = x  # Save for skip connection
        x = self.pool1(x)
        
        # Encoder block 2
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        skip2 = x  # Save for skip connection
        x = self.pool2(x)
        
        # Encoder block 3
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        skip3 = x  # Save for skip connection
        x = self.pool3(x)
        
        # Encoder block 4
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = F.relu(self.conv8(x))
        x = self.dropout(x)
        
        return x, (skip1, skip2, skip3)


class Decoder(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(Decoder, self).__init__()
        
        # Decoder layers
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.PixelShuffle(2)
        
        self.conv3 = nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.PixelShuffle(2)
        
        self.conv5 = nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.up3 = nn.PixelShuffle(2)
        
        self.conv7 = nn.Conv2d(32 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # Denoising score
        self.conv9 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # Uncertainty variance
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
    def forward(self, x, skip_connections):
        skip1, skip2, skip3 = skip_connections
        
        # Decoder block 1
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.up1(x)
        
        # Decoder block 2
        x = torch.cat((x, skip3), dim=1)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.up2(x)
        
        # Decoder block 3
        x = torch.cat((x, skip2), dim=1)
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = self.up3(x)
        
        # Decoder block 4
        x = torch.cat((x, skip1), dim=1)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        
        # Output layers
        denoising_score = self.conv8(x)
        uncertainty_variance = self.conv9(x)
        
        return denoising_score, uncertainty_variance


class UncertaintyScoreEstimationNetwork(nn.Module):
    def __init__(self, input_channels=2, dropout_rate=0.0):
        super(UncertaintyScoreEstimationNetwork, self).__init__()
        
        # Encoder
        self.encoder = Encoder(input_channels, dropout_rate)
        
        # Decoder
        self.decoder = Decoder(dropout_rate)
        
    def forward(self, noisy_image, denoised_image):
        # Concatenate noisy image and denoised image along the channel dimension
        concatenated_input = torch.cat((noisy_image, denoised_image), dim=1)
        
        # Pass through encoder
        encoded_features, skip_connections = self.encoder(concatenated_input)
        
        # Pass through decoder
        denoising_score, uncertainty_variance = self.decoder(encoded_features, skip_connections)
        
        return uncertainty_variance, denoising_score


class Network(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(Network, self).__init__()
        self.fcn = FCN(dropout_rate)
        self.unet = UNet(dropout_rate)
        self.Uncertaintymodel = UncertaintyScoreEstimationNetwork(input_channels=2, dropout_rate=dropout_rate)
    
    def forward(self, x):
        y = x
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x  # out:去噪后的图像
        
        score_map, variance_map = self.Uncertaintymodel(y, out)
        
        return noise_level, out, score_map, variance_map


class fixed_loss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta  # 不确定性权重系数

    def forward(self, out_image, gt_image, est_noise, gt_noise, variance_map, if_asym):
        l2_loss = F.mse_loss(out_image, gt_image)
        uncertainty_loss = torch.mean(torch.exp(-variance_map) * l2_loss + self.beta * variance_map)
        asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))

        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = l2_loss + 0.5 * asym_loss + 0.05 * tvloss + uncertainty_loss

        return loss

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]


