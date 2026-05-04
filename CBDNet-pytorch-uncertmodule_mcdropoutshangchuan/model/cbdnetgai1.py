
import torch
import torch.nn as nn
import torch.nn.functional as F


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)#修改

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
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

#nn.Conv2d(3, 32, 3, padding=1):
#第一个参数3：表示输入图像的通道数（channel）。对于彩色RGB图像，通道数通常是3（红、绿、蓝）。
#第二个参数32：表示输出通道数（也称为卷积核或滤波器的数量）。这意味着该卷积层将学习32个不同的卷积核，每个卷积核都会生成一个输出特征图（feature map）。
#第三个参数3：表示卷积核的大小（kernel size），这里是3x3。卷积核是一个小的权重矩阵，它会在输入图像上滑动，进行点积运算以提取特征。
#这个类用于（提取噪声特征）预测噪声水平。
#这个类的通道数进行了修改
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.fcn(x)

#这个类的通道进行了修改
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.inc = nn.Sequential(
            single_conv(2, 32),
            single_conv(32, 32)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(32, 64),
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up1 = up(128)
        self.conv3 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.up2 = up(64)
        self.conv4 = nn.Sequential(
            single_conv(32, 32),
            single_conv(32, 32)
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


class UncertaintyScoreEstimationNetwork(nn.Module):
    def __init__(self, input_channels=2):
        super(UncertaintyScoreEstimationNetwork, self).__init__()
        
        # Encoder
        self.encoder = Encoder(input_channels)
        
        # Decoder
        self.decoder = Decoder()
        
    def forward(self, noisy_image, denoised_image):
        # Concatenate noisy image and denoised image along the channel dimension
        concatenated_input = torch.cat((noisy_image, denoised_image), dim=1)
        
        # Pass through encoder
        encoded_features, skip_connections = self.encoder(concatenated_input)
        
        # Pass through decoder
        denoising_score, uncertainty_variance = self.decoder(encoded_features, skip_connections)
        
        return uncertainty_variance, denoising_score

class Encoder(nn.Module):
    def __init__(self, input_channels):
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
        
        self.dropout = nn.Dropout2d(p=0.5)
        
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
    def __init__(self):
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
        
        self.dropout = nn.Dropout2d(p=0.5)
        
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
'''
class UncertaintyScoreEstimationNetwork(nn.Module):
    def __init__(self, in_channels=2, num_conv_layers=8, num_dropout_layers=8, num_pool_layers=3):
        super(UncertaintyScoreEstimationNetwork, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            *[nn.Conv2d(in_channels if i == 0 else num_conv_layers, num_conv_layers, kernel_size=3, padding=1)
              for i in range(num_conv_layers)]
            +[nn.Dropout(0.5) for _ in range(num_dropout_layers)]
        )
        
        # Pooling Down-sampler
        self.pool_down_sampler = nn.Sequential(
            *[nn.AvgPool2d(kernel_size=2, stride=2) for _ in range(num_pool_layers)]
        )
        
        # Decoder (assuming symmetric structure for simplicity)
        self.decoder = nn.Sequential(
            *[nn.ConvTranspose2d(num_conv_layers, num_conv_layers, kernel_size=2, stride=2) for _ in range(num_pool_layers)],
            *[nn.Conv2d(num_conv_layers, num_conv_layers, kernel_size=3, padding=1) for _ in range(num_conv_layers)]
        )
        
        # Up-sampler (additional convolution for fine-tuning the output)
        self.up_sampler = nn.Conv2d(num_conv_layers * 2, num_conv_layers, kernel_size=1)  # Assuming concatenation of features
        
        # Final output layers (assuming two outputs: uncertainty variance and denoising score)
        self.uncertainty_variance_output = nn.Conv2d(num_conv_layers, 1, kernel_size=1)
        self.denoising_score_output = nn.Conv2d(num_conv_layers, 1, kernel_size=1)
        
        # Jump connections (assuming skip connections from every other convolution layer in the encoder)
        self.jump_connections = nn.ModuleList([
            nn.Conv2d(num_conv_layers, num_conv_layers, kernel_size=1) for _ in range(0, num_conv_layers, 2)
        ])
        
        # Adjust input channels for jump connections (if needed)
        self.adjust_jump_input = nn.Conv2d(in_channels, num_conv_layers, kernel_size=1) if in_channels != num_conv_layers else nn.Sequential()
        
    def forward(self, noisy_image, denoised_image):
        # Combine noisy image and denoised image (assuming concatenation along the channel dimension)
        combined_image = torch.cat((noisy_image, denoised_image), dim=1)
        
        # Adjust input channels for the first convolution
        combined_image = self.adjust_jump_input(combined_image)
        
        # Encoder path
        encoder_features = []
        x = combined_image
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i % 2 == 0 and i < len(self.jump_connections):  # Save features for jump connections
                encoder_features.append(x)
        
        # Pooling down-sampling
        x = self.pool_down_sampler(x)
        
        # Decoder path
        for i, (pool_layer, conv_layer) in enumerate(zip(reversed(self.pool_down_sampler), reversed(self.decoder[:len(self.pool_down_sampler)]))):
            x = conv_layer(F.interpolate(x, scale_factor=2, mode='nearest'))  # Up-sampling
            if i < len(encoder_features):
                jump_feature = encoder_features[-(i+1)]
                jump_feature = F.interpolate(jump_feature, size=x.size()[2:], mode='nearest')  # Match spatial dimensions
                jump_feature = self.jump_connections[-(i+1)](jump_feature)
                x = torch.cat((x, jump_feature), dim=1)  # Concatenate features
        
        # Remaining decoder convolutions
        for layer in self.decoder[len(self.pool_down_sampler):]:
            x = layer(x)
        
        # Concatenate with initial encoder features (assuming we want to use all encoder features for fine-tuning)
        # (In practice, this may be adjusted based on the actual network design)
        x = torch.cat((x, combined_image), dim=1)
        
        # Final up-sampling convolution (to adjust channel dimensions)
        x = self.up_sampler(x)
        
        # Output layers
        uncertainty_variance = self.uncertainty_variance_output(x)
        denoising_score = self.denoising_score_output(x)
        
        return uncertainty_variance, denoising_score
'''

'''
class UncertaintyScoreEstimationNetwork(nn.Module):
    def __init__(self):
        super(UncertaintyScoreEstimationNetwork, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),  # Input is concatenation of noisy and denoised image, so in_channels=2
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),  # First downsampling
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Second downsampling
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Third downsampling
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),  # First upsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),  # Second upsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)  # Output two channels: uncertainty variance and denoising score
        )
        
        
    def forward(self, noisy_image, denoised_image):
        # Concatenate noisy image and denoised image along the channel dimension
        input_tensor = torch.cat((noisy_image, denoised_image), dim=1)
        print("----------input_tensor---------",input_tensor.shape) #torch.Size([8, 2, 96, 96])
        # Encode the input tensor
        encoded_features = self.encoder(input_tensor)
        
        # Decode the encoded features to get uncertainty variance and denoising score
        output_tensor = self.decoder(encoded_features)
        print("----------output_tensor---------",output_tensor.shape)   #torch.Size([8, 2, 48, 48])
        # Split the output tensor into uncertainty variance and denoising score
        uncertainty_variance, denoising_score = torch.split(output_tensor, 1, dim=1)
        
        return uncertainty_variance, denoising_score
 '''   
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()
        self.Uncertaintymodel = UncertaintyScoreEstimationNetwork()    #不确定性得分估计-------------
    
    def forward(self, x):
        y=x
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x #out:去噪后的图像
        
        # print("------out :----", out.shape)  #torch.Size([1, 1, 128, 128])
        # print("------y :----", y.shape)    #torch.Size([1, 1, 128, 128])
        
        score_map, variance_map = self.Uncertaintymodel(y, out)    ##不确定性得分和方差,out:denoisedimg, y:noisyimg-------
        print("------Score Map Shape:----", score_map.shape)  #不确定性得分估计------------- torch.Size([1, 1, 64, 64])
        print("-----Variance Map Shape:----", variance_map.shape)    #不确定性方差-----------torch.Size([1, 1, 64, 64])
        print("------out :----", out .shape)  #torch.Size([1, 1, 128, 128])
        print("------x :----", x .shape)    ##torch.Size([1, 1, 128, 128]),但最原始的openneuro尺寸为240*240
        
        return noise_level, out, score_map, variance_map


class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    ##criterion(去噪后的图，干净图，估计的sigma，真实的sigma,flag)
    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        l2_loss = F.mse_loss(out_image, gt_image)
        #calibration_loss=F.mse_loss(score_map,gt_image)
        #uncertainty_loss = torch.mean(torch.exp(-score_map) * l2_loss + self.beta * score_map)
        #模型输出的noise_level_est, sigma_var对应损失函数的第三、四个参数：est_noise, gt_noise。下面查查sigma_var来自数据loader的数据集syn里的sigma_img图像
        asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))

        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = l2_loss +  0.5 * asym_loss + 0.05 * tvloss

        return loss

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]