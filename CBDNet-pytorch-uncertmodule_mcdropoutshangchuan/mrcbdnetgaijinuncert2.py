import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

# ==================== 配置参数 ====================
class Config:
    clean_root = r"/mnt/gemlab_data/Medical_image_database/MRI/denoise/data2mropenneuro/cleanimg_train/"
    noisy_root = r"/mnt/gemlab_data/Medical_image_database/MRI/denoise/data2mropenneuro/noisyimg_train/"
    modalities = ['t1', 't2', 'flair','t1ce']  # 支持的模态类型
    batch_size = 8                       # 实际每个GPU处理2个样本
    num_workers = 8
    lr = 1e-4
    epochs = 50
    mc_dropout_samples = 10             # 蒙特卡洛采样次数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
config = Config()

# ==================== 数据加载 ====================
class MRIDataset3D(Dataset):
    def __init__(self, clean_root, noisy_root):
        self.subjects = sorted(os.listdir(clean_root))
        self.clean_files, self.noisy_files = self._parse_pairs(clean_root, noisy_root)
        
    def _parse_pairs(self, clean_root, noisy_root):
        file_pairs = []
        for subj in self.subjects:
            clean_dir = os.path.join(clean_root, subj)
            noisy_dir = os.path.join(noisy_root, subj)
            
            # 收集所有模态的配对文件
            for mod in config.modalities:
                clean_path = os.path.join(clean_dir, f"{subj}_{mod}.nii.gz")
                noisy_path = os.path.join(noisy_dir, f"{subj}_{mod}.nii.gz")
                if os.path.exists(clean_path) and os.path.exists(noisy_path):
                    file_pairs.append((clean_path, noisy_path))
        return file_pairs
    
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # 加载数据并保留affine信息
        clean_nii = nib.load(self.clean_files[idx])
        noisy_nii = nib.load(self.noisy_files[idx])
        
        clean = torch.tensor(clean_nii.get_fdata(), dtype=torch.float32).unsqueeze(0)  # [1, H, W, D]
        noisy = torch.tensor(noisy_nii.get_fdata(), dtype=torch.float32).unsqueeze(0)
        
        return {
            'clean': clean,
            'noisy': noisy,
            'affine': clean_nii.affine,
            'modality': os.path.basename(self.clean_files[idx]).split('_')[-1].split('.')
        }

# ==================== 改进的3D CBDNet ====================
class UncertaintyConv3D(nn.Module):
    """不确定性估计卷积块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.Dropout3d(0.5),  # 蒙特卡洛Dropout
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x)

class NoiseEstimationSubnet3D(nn.Module):
    """3D噪声估计子网络"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            UncertaintyConv3D(1, 32),
            nn.MaxPool3d(2),
            UncertaintyConv3D(32, 64),
            nn.MaxPool3d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(32, 1, 3, padding=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

class CBDNet3D(nn.Module):
    """改进的3D CBDNet"""
    def __init__(self):
        super().__init__()
        self.noise_subnet = NoiseEstimationSubnet3D()
        self.denoise_subnet = nn.Sequential(
            UncertaintyConv3D(1, 32),
            UncertaintyConv3D(32, 64),
            UncertaintyConv3D(64, 32),
            nn.Conv3d(32, 1, 3, padding=1)
        )
        
    def forward(self, x):
        noise = self.noise_subnet(x)
        denoised = self.denoise_subnet(x - noise)
        return denoised, noise

# ==================== 不确定性校准损失 ====================
class UncertaintyAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, preds, clean, uncertainty):
        # preds: [B, 1, D, H, W]
        # uncertainty: [B, 1, D, H, W]
        base_loss = self.mse(preds, clean)
        weighted_loss = base_loss / (2 * torch.exp(uncertainty)) + 0.5 * uncertainty
        return weighted_loss.mean()

# ==================== 训练引擎 ====================
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.criterion = UncertaintyAwareLoss()
        self.train_loader, self.val_loader = self._create_loaders()
        self.best_psnr = 0.0
        
    def _create_loaders(self):
        dataset = MRIDataset3D(config.clean_root, config.noisy_root)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=config.batch_size, 
                                shuffle=True, num_workers=config.num_workers)
        val_loader = DataLoader(val_set, batch_size=config.batch_size,
                               num_workers=config.num_workers)
        return train_loader, val_loader
    
    def _train_epoch(self, epoch):
        self.model.train()
        losses = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            noisy = batch['noisy'].to(config.device)
            clean = batch['clean'].to(config.device)
            
            self.optimizer.zero_grad()
            denoised, noise = self.model(noisy)
            loss = self.criterion(denoised, clean, noise)
            
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(losses[-10:])})
            
        return np.mean(losses)
    
    def _validate(self):
        self.model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                noisy = batch['noisy'].to(config.device)
                clean = batch['clean'].cpu().numpy()
                
                # 蒙特卡洛采样计算不确定性
                denoised_samples = []
                for _ in range(config.mc_dropout_samples):
                    denoised, _ = self.model(noisy)
                    denoised_samples.append(denoised.cpu().numpy())
                
                denoised_mean = np.mean(denoised_samples, axis=0)
                denoised_std = np.std(denoised_samples, axis=0)
