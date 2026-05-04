import os, time, scipy.io, shutil
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2

from model.cbdnet import Network
from utils import read_img, chw_to_hwc, hwc_to_chw
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = 'Test')
parser.add_argument('input_filename', type=str)
parser.add_argument('output_filename', type=str)
args = parser.parse_args()

input_folder = "/mnt/gemlab_data/Medical_image_database/MRI/segmentation/jmf-result/prostate_origin_8ProstateX/imagesTr/" 
    
pathfile="/mnt/gemlab_data/Medical_image_database/MRI/segmentation/jmf-result/prostate_denoised_nii_uncertainty/"        
	
output_folder = os.path.join(pathfile,"qiepian") 
# 如果output_folder不存在，则创建  
if not os.path.exists(output_folder):  
	os.makedirs(output_folder) 
save_dir = './save_model/'

model = Network()
model.cuda()
model = nn.DataParallel(model)

model.eval()

if os.path.exists(os.path.join(save_dir, 'uncertainty2000epochcheckpoint.pth.tar')):
    # load existing model
    model_info = torch.load(os.path.join(save_dir, 'uncertainty2000epochcheckpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
else:
    print('Error: no trained model detected!')
    exit(1)

files = os.listdir(input_folder) 
files.sort() #'BraTS2021_00000', 'BraTS2021_00002', 'BraTS2021_00003', 'BraTS2021_00005', ..........
print("----files----",files) 
for f in files:
        path=os.path.join(input_folder,f)
        files2 = os.listdir(path)
        nii_files = [f2 for f2 in files2 if f2.endswith(".nii.gz")]    
        print("----origin_nii_files----", nii_files) 
        
        
        # 创建一个空字典来存储每个模态切片  
        modalities = {"adc": [],"t2w": []}  
        # 读取每个nii.gz文件并将其切片添加到相应模态的字典中 
        for file in nii_files:  
            # 正则表达式模式 ，匹配4个模态的文件名字
            parts = file.split('-')  
            # 因为我们想要的是前两个部分，所以我们可以将它们合并  
            two_string = '-'.join(parts[:2])  
            pattern = r'\-([a-z]+[0-9]+[a-z]+|[a-z]+[0-9]+|[a-z]+)(?:\.nii\.gz|\.nii)$'  
            # 获取模态类型  
            match = re.search(pattern, file)    
            if match:    
                modality = match.group(1)    
            else:    
                print(f"Cannot determine modality from filename {file}")    
                continue  # 跳过当前文件  
            
            image = sitk.ReadImage(os.path.join(path, file))  
            shape = image.GetSize()  
            print(f"Shape of {file}: {shape}")  
            
            # 将图像转换为灰度图  
            gray_image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)  
            
            # 返回灰度图像数据  
            #dataorg = sitk.GetArrayFromImage(gray_image) 
            #data = np.transpose(dataorg, (1, 2, 0))  #brats2021用这行
            data = sitk.GetArrayFromImage(gray_image) 
                        
            print("----data.shape----",data.shape) 
            # 获取切片数量并遍历每个切片  
            num_slices = data.shape[2]  
            
            i=0
            for i in range(num_slices):  
                # 获取切片数据并将其转换为灰度图像  
                slice_data = data[:,:,i]
                #slice_data = np.flip(slice_data, axis=(0, 1)) #图像旋转180度，在openneuro使用，其他数据集本行屏蔽  
                slice_data = np.flipud(slice_data) #前后翻转180度，prostate时用其他数据集屏蔽
                slice_img = Image.fromarray(slice_data).convert("L")  

                # 将切片保存为jpg图像，并按照指定的命名格式进行命名  
                slice_name = "{}-{:05d}-{}.jpg".format(two_string, i + 1, modality)
                output_qiepian = os.path.join(output_folder,two_string)
                # 如果output_folder不存在，则创建  
                if not os.path.exists(output_qiepian):  
                    os.makedirs(output_qiepian) 
                slice_img.save(os.path.join(output_qiepian, slice_name))  
            
                # 将切片添加到相应模态的字典中  
                modalities[modality].append(slice_name)   
                
		# 打印处理完成的消息以及每个模态的切片数量  
        print("处理完成！")  
        for modality, slices in modalities.items():  
            print("{}模态共有{}个切片.".format(modality, len(slices)))
        
        # 获取相同切片序号的所有图片  
        image_files = os.listdir(output_qiepian)  
        image_files.sort()  #给放到listdir里的out_folder文件排序
        #print("----------------image_files.sort()------------",image_files)    #排序为：flair,t1,t1ce,t2
        same_slice_images = {}  # 用字典存储相同切片序号的图像文件名列表，键为切片序号，值为文件名列表  
        for file in image_files:  
            if file.endswith(".jpg"):  # 只处理jpg文件
                parts = file.split('-')  
                slice_number = int(parts[2])   # 提取切片序号 
                same_slice_images.setdefault(slice_number, []).append(os.path.join(output_qiepian, file))  # 将图片文件名添加到列表中  
        
        # 不同切片序号的图像读取去噪  
        for slice_number, image_files in same_slice_images.items():  
            images = []  
            for file in image_files:  
                image = Image.open(file).convert("L")  # 转换为灰度图，如果需要其他颜色空间请修改此处参数  
                images.append(np.array(image))  # 将PIL图像转换为numpy数组，以便后续处理  
            # 使用torch.cat将四个模态的图像数据合并为一个4通道的图像张量，并保存为文件  
            for img in images  
    			#input_image = read_img(args.input_filename)
				input_image =img
				input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()

				with torch.no_grad():
					noise_level_est, output, score_map, variance_map = model(input_var)


				#可视化不确定性图和方差
				score_mapx = score_map  # 随机生成示例数据
				variance_mapx = variance_map.abs()  # 随机生成示例数据，并确保方差是非负的

				# 为了可视化，我们需要将 torch.Tensor 转换为 numpy.ndarray
				score_map_np = score_mapx.detach().cpu().numpy()
				variance_map_np = variance_mapx.detach().cpu().numpy()

				# 设置保存图像的文件夹路径
				save_dir = r"/mnt/gemlab_data/Medical_image_database/MRI/segmentation/jmf-result/prostate_denoised_uncertmap_uncertainty/"
				os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在，如果不存在则创建

				
    
    
    
    			# 定义图像大小和边距,大小应该与原图像一致。。。。。。
				fig_size = (100, 100)
				margin = 0.1  # 用于颜色条的边距

				# 遍历所有批次并进行可视化及保存
				for batch_index in range(score_map_np.shape[0]):
					# 移除批次和通道维度，只保留高度和宽度
					score_map_batch = score_map_np[batch_index, 0, :, :]
					variance_map_batch = variance_map_np[batch_index, 0, :, :]

					# 可视化不确定性得分估计并保存
					fig, ax = plt.subplots(figsize=fig_size)
					cax = ax.inset_axes([1.02, 0.2, 0.05, 0.6])  # 在右侧添加颜色条轴
					im = ax.imshow(score_map_batch, cmap='viridis')
					fig.colorbar(im, cax=cax)
					ax.set_title(f'Batch {batch_index + 1}: Uncertainty Score Map')
					ax.axis('off')  # 移除坐标轴
					score_map_filename = os.path.join(save_dir, f'score_map_batch_{batch_index + 1}_with_colorbar.png')
					plt.savefig(score_map_filename, bbox_inches='tight', pad_inches=0)
					plt.close()

					# 可视化不确定性方差并保存
					fig, ax = plt.subplots(figsize=fig_size)
					cax = ax.inset_axes([1.02, 0.2, 0.05, 0.6])  # 在右侧添加颜色条轴
					im = ax.imshow(variance_map_batch, cmap='hot')
					fig.colorbar(im, cax=cax)
					ax.set_title(f'Batch {batch_index + 1}: Uncertainty Variance Map')
					ax.axis('off')  # 移除坐标轴
					variance_map_filename = os.path.join(save_dir, f'variance_map_batch_{batch_index + 1}_with_colorbar.png')
					plt.savefig(variance_map_filename, bbox_inches='tight', pad_inches=0)
					plt.close()




				output_image = chw_to_hwc(output[0,...].cpu().numpy())
				output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

				cv2.imwrite(args.output_filename, output_image)