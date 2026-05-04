import os, time, scipy.io, shutil
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
from torchvision import utils

#from model.cbdnetorigin import Network
from model.cbdnet import Network  #uncertainty
from utils import read_img, chw_to_hwc, hwc_to_chw
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description = 'Test')
parser.add_argument('input_filename', type=str)
parser.add_argument('output_filename', type=str)
args = parser.parse_args()

save_dir = './save_model/'

model = Network(dropout_rate=0.0).to(device)
model.cuda()
model = nn.DataParallel(model)

model.eval()

if os.path.exists(os.path.join(save_dir, 'uncertmodule_mcdropout_3000checkpoint.pth.tar')):
    # load existing model
    model_info = torch.load(os.path.join(save_dir, 'uncertmodule_mcdropout_3000checkpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
else:
    print('Error: no trained model detected!')
    exit(1)

input_image = read_img(args.input_filename)
print("---------input_image------",input_image.shape)   # (240, 240, 1)
input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()
print("---------input_var------",input_var.shape)   # torch.Size([1, 1, 240, 240])

with torch.no_grad():
    for _ in range(200):
        predictions = []
        noise_level_est, output, score_map, variance_map = model(input_var)
        predictions.append(output)
    mean_pred = torch.mean(torch.stack(predictions), dim=0)
    uncertainty = torch.var(torch.stack(predictions), dim=0)

#keyiyong de 
# with torch.no_grad():
#     noise_level_est, output, score_map, variance_map = model(input_var)

print("-----uncertainty----------",mean_pred.shape)
#可视化不确定性图和方差===============不带颜色条===========，直接保存为与原图像等尺寸的jpg格式
# 创建一个保存图像的目录（如果不存在）
save_dir2 = './Test_Image/uncertaintymap'
os.makedirs(save_dir2, exist_ok=True)
 
# 指定保存路径和文件名
save_pathsco = os.path.join(save_dir2, "imagescorebrats21_71_t2.jpg")
save_pathvar = os.path.join(save_dir2, "imagevarbrats21_71_t2.jpg")
save_pathvarxin = os.path.join(save_dir2, "image00071mentoclo.jpg")
 
# 使用 torchvision.utils.save_image 保存图像
# 注意：save_image 默认会将图像归一化到 [0, 1] 范围，并添加 batch 和 channel 维度
# 这里因为我们的张量已经是 [1, 1, 64, 64] 格式，所以直接保存即可
utils.save_image(score_map, save_pathsco, normalize=True, nrow=1, padding=0)
utils.save_image(variance_map, save_pathvar, normalize=True, nrow=1, padding=0)
utils.save_image(uncertainty, save_pathvarxin, normalize=True, nrow=1, padding=0)

#保存save_pathvarxin，   uncertainty -----------
# 提取第一个通道的数据并去除批次维度
# 结果形状为 [240, 240]
image_data = mean_pred[0, 0].detach().cpu().numpy()
# 将数据归一化到 [0, 255] 范围并转换为 uint8 类型
# 注意：如果数据已经是 [0, 1] 范围，可以直接乘以 255
# 如果数据是任意范围，需要先归一化
image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
image_data = image_data.astype(np.uint8)
# 使用 OpenCV 保存为 JPG
cv2.imwrite('Test_Image/output/00071_2_t1_uncertainty3000epoch.jpg', image_data)

 
print(f"图像已保存到 {save_pathsco}")



output_image = chw_to_hwc(output[0,...].cpu().numpy())
output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

cv2.imwrite(args.output_filename, output_image)