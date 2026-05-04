import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt

# ==================== 数据加载部分 ====================
class PairedMRIDataset(Dataset):
    def __init__(self, clean_root, noisy_root, cases):
        """
        clean_root: 干净图像根目录 
        noisy_root: 含噪图像根目录
        cases: 病例ID列表 (openneuro_01667 到 openneuro_01688)
        """
        self.pairs = []
        
        # 遍历所有病例
        for case in cases:
            clean_case_path = os.path.join(clean_root, case)
            noisy_case_path = os.path.join(noisy_root, case)
            
            # 获取该病例所有模态文件
            for fname in os.listdir(clean_case_path):
                if fname.endswith('.nii.gz'):
                    # 构建配对路径
                    clean_path = os.path.join(clean_case_path, fname)
                    noisy_path = os.path.join(noisy_case_path, fname)
                    
                    # 验证文件存在性
                    if os.path.exists(noisy_path):
                        self.pairs.append((clean_path, noisy_path))
                    else:
                        print(f"警告：缺失配对文件 {noisy_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[idx]
        
        # 加载nii数据
        clean_img, affine = load_nii(clean_path)
        noisy_img, _ = load_nii(noisy_path)
        
        # 数据预处理
        clean_tensor = preprocess(clean_img)
        noisy_tensor = preprocess(noisy_img)
        
        return noisy_tensor, clean_tensor, affine

def load_nii(file_path):
    """加载NIfTI文件并返回数据和affine矩阵"""
    nii = nib.load(file_path)
    return nii.get_fdata().astype(np.float32), nii.affine

def preprocess(data):
    """数据预处理：归一化 + 维度扩展"""
    # 归一化到[0,1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    # 添加通道和批次维度 [C, D, H, W]
    return torch.tensor(data[np.newaxis, ...]) 

# ==================== 模型定义 ====================
class CBDNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # 噪声估计子网络
        self.noise_estimation = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, padding=1)
        )
        
        # 去噪子网络（包含不确定性通道）
        self.denoiser = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 2, kernel_size=3, padding=1)  # 输出含不确定性通道
        )

    def forward(self, x):
        # 噪声估计
        noise_level = self.noise_estimation(x)
        
        # 拼接噪声估计和输入
        combined = torch.cat([x, noise_level], dim=1)
        
        # 去噪输出（主输出 + 不确定性）
        output = self.denoiser(combined)
        return output[:, 0:1], output[:, 1:2], noise_level  # 返回去噪结果、不确定性和噪声估计

# ==================== 训练相关 ====================
class UncertaintyAwareLoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta  # 不确定性权重系数
        self.mse = nn.MSELoss()
        
    def forward(self, pred, uncertainty, target):
        # 基础重建损失
        recon_loss = self.mse(pred, target)
        
        # 不确定性校准损失
        uncertainty_loss = torch.mean(torch.exp(-uncertainty) * recon_loss + self.beta * uncertainty)
        
        return recon_loss + uncertainty_loss

def train_loop(model, loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (noisy, clean, _) in enumerate(loader):
        noisy = noisy.to(device,non_blocking=True)
        clean = clean.to(device,non_blocking=True)
        
        optimizer.zero_grad()
        
        # 前向传播
        pred, uncertainty, noise_est = model(noisy)
        
        # 计算损失
        loss = criterion(pred, uncertainty, clean)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 每10个batch打印一次
        if batch_idx % 10 == 0:
            print(f'Train Batch: {batch_idx}/{len(loader)} Loss: {loss.item():.4f}')

# ==================== 评估与可视化 ====================
def evaluate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    
    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy = noisy.to(device)
            clean = clean.numpy().squeeze()
            
            # 预测
            pred, _, _ = model(noisy)
            pred = pred.cpu().numpy().squeeze()
            
            # 计算指标
            total_psnr += psnr(clean, pred, data_range=1.0)
            total_ssim += ssim(clean, pred, multichannel=False, data_range=1.0)
    
    return total_psnr/len(loader), total_ssim/len(loader)

def visualize_results(noisy, pred, clean, uncertainty, slice_idx=None):
    """可视化中间切片结果"""
    plt.figure(figsize=(15, 10))
    
    # 自动选择中间切片
    if slice_idx is None:
        slice_idx = pred.shape[-1] // 2
    
    # 显示各图像
    plt.subplot(141)
    plt.imshow(noisy[0,0,:,:,slice_idx], cmap='gray')
    plt.title('Noisy Input')
    
    plt.subplot(142)
    plt.imshow(pred[0,0,:,:,slice_idx], cmap='gray')
    plt.title('Denoised')
    
    plt.subplot(143)
    plt.imshow(clean[0,0,:,:,slice_idx], cmap='gray')
    plt.title('Clean Target')
    
    plt.subplot(144)
    plt.imshow(uncertainty[0,0,:,:,slice_idx], cmap='viridis')
    plt.title('Uncertainty')
    plt.colorbar()
    
    plt.show()

# ==================== 主执行流程 ====================
if __name__ == "__main__":
    # 配置参数
    print(f"可用 GPU 列表: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.current_device()}")
    
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 4, "需要至少 4 个 GPU"

    # 主设备设为第一个 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cases = [f"openneuro_{i:05d}" for i in range(1667, 1689)]  # 生成所有病例ID
    
    # 初始化数据集
    dataset = PairedMRIDataset(
        clean_root=r"/mnt/gemlab_data/Medical_image_database/MRI/denoise/data2mropenneuro/cleanimg_train/",
        noisy_root=r"/mnt/gemlab_data/Medical_image_database/MRI/denoise/data2mropenneuro/noisyimg_train/",
        cases=cases
    )
    
    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)
    
    # 初始化模型
    #model = CBDNet3D().to(device)
    model = CBDNet3D()
    model = nn.DataParallel(model)  # 关键代码
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = UncertaintyAwareLoss(beta=0.6)
    
    # 训练循环
    for epoch in range(30):
        print(f"\nEpoch {epoch+1}")
        train_loop(model, loader, optimizer, criterion, device)
        #在训练循环中添加GPU内存监控
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        # 每5个epoch评估一次
        if (epoch+1) % 5 == 0:
            avg_psnr, avg_ssim = evaluate(model, loader, device)
            print(f"Validation PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
            
            # 可视化示例结果
            sample = next(iter(loader))
            noisy, clean, affine = sample
            with torch.no_grad():
                pred, uncertainty, _ = model(noisy.to(device))
            visualize_results(noisy.numpy(), pred.cpu().numpy(), clean.numpy(), 
                            uncertainty.cpu().numpy())
    
    # 保存最终模型
    torch.save(model.state_dict(), "3d_cbdnet_final.pth")
