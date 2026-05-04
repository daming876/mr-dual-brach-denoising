#去噪后保存为nii格式
import os, time, scipy.io, shutil
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import re

from model.cbdnet import Network
from utils import read_img, chw_to_hwc, hwc_to_chw
import matplotlib.pyplot as plt
import SimpleITK as sitk 
from PIL import Image 

# parser = argparse.ArgumentParser(description = 'Test')
# parser.add_argument('input_filename', type=str)
# parser.add_argument('output_filename', type=str)
# args = parser.parse_args()

input_folder = "/mnt/gemlab_data_2/jiang/CBDNet/prostate_origin_8ProstateX_files/"  #文件夹里是类似nii.gz文件ProstateX-0203-t2w.nii.gz
    
pathfile="/mnt/gemlab_data_2/jiang/CBDNet/prostate_denoised_nii/"        

output_folder = os.path.join(pathfile,"noisyqiepian") 
output_folder2 = os.path.join(pathfile,"calibration_denoised3000epochs") 
# 如果output_folder不存在，则创建  
if not os.path.exists(output_folder):  
	os.makedirs(output_folder) 
 
if not os.path.exists(output_folder2):  
	os.makedirs(output_folder2)

save_dir = './save_model/'

model = Network()
model.cuda()
model = nn.DataParallel(model)

model.eval()

if os.path.exists(os.path.join(save_dir, 'calibration3000checkpoint.pth.tar')):
    # load existing model
    model_info = torch.load(os.path.join(save_dir, 'calibration3000checkpoint.pth.tar'))
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
                        
            print("-------data.shape--------",data.shape) 
            # 获取切片数量并遍历每个切片  
            num_slices = data.shape[0]  
            
            i=0
            for i in range(num_slices):  
                # 获取切片数据并将其转换为灰度图像  
                slice_data = data[i,:,:]
                #slice_data = np.flip(slice_data, axis=(0, 1)) #图像旋转180度，在openneuro使用，其他数据集本行屏蔽  
                slice_data = np.flipud(slice_data) #前后翻转180度，prostate时用，其他数据集屏蔽
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
            print("{}模态共有{}个切片.".format(modality, len(slices)))#32个切片，每个nii：（32，160，160）







# 设置路径
# input_dir = '/mnt/gemlab_data_2/jiang/CBDNet/prostate_denoised_nii/noisyqiepian/'
# output_dir = '/mnt/gemlab_data_2/jiang/CBDNet/prostate_denoised_nii/denoisedqiepian/'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 假设您的去噪模型接受归一化到[0, 1]的RGB图像作为输入
# # 并输出相同大小的图像
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 转换为Tensor，自动归一化到[0, 1]
#     transforms.Lambda(lambda x: x.unsqueeze(0))  # 增加batch维度
# ])

# # 加载去噪模型
# model = YourDenoiseModel().to(device)
# model.load_state_dict(torch.load('path_to_your_model.pth'))  # 加载模型权重
# model.eval()

# 遍历输入文件夹中的每个子文件夹
for subfolder_name in os.listdir(output_folder):
    subfolder_path = os.path.join(output_folder, subfolder_name)
    if os.path.isdir(subfolder_path):
        output_subfolder_path = os.path.join(output_folder2, subfolder_name)
        os.makedirs(output_subfolder_path, exist_ok=True)
        
        # 遍历子文件夹中的每个图像文件
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                file_path = os.path.join(subfolder_path, filename)
                
                # 读取图像
                #image = Image.open(file_path)
                input_image = read_img(file_path)
                #input_image =image.astype('float32')
                print("-------input_image---",input_image.shape) #(160, 160, 1)
                input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()
                #input_var =  torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).cuda()
                print("-------input_var--",input_var.shape) # torch.Size([1, 1, 160, 160])

                with torch.no_grad():
                    noise_level_est, output, score_map, variance_map = model(input_var)
                    
                # # 将处理后的图像转换回PIL Image格式
                # denoised_image = Image.fromarray((denoised_image_tensor * 255).astype(np.uint8))
                print("-------noise_level_est--",noise_level_est.shape) #torch.Size([1, 1, 160, 160])
                print("-------output--",output.shape) #torch.Size([1, 1, 160, 160])
                print("-------score_map--",score_map.shape) #torch.Size([1, 1, 160, 80])
                print("-------variance_map--",variance_map.shape) #torch.Size([1, 1, 160, 160])
                #可视化不确定性图和方差
                #score_mapx = score_map  # 随机生成示例数据
                variance_mapx = variance_map.abs()  # 随机生成示例数据，并确保方差是非负的

                
                score_map_image = chw_to_hwc(score_map[0,...].cpu().numpy())
                score_map_image = np.uint8(np.round(np.clip(score_map_image, 0, 1) * 255.))[: ,: ,::-1]

                variance_map_image = chw_to_hwc(variance_mapx[0,...].cpu().numpy())
                variance_map_image = np.uint8(np.round(np.clip(variance_map_image, 0, 1) * 255.))[: ,: ,::-1]

                output_image = chw_to_hwc(output[0,...].cpu().numpy())
                output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]
                
                # parts = filename.split('-')
                # four_string = '-'.join(parts[:3]) 
                
                parts = filename.split('.')
                four_string = '.'.join(parts[:1]) 
                
                
                filename_score = "{}-score.jpg".format(four_string)
                # 保存处理后的图像
                output_file_path1 = os.path.join(output_subfolder_path, filename_score)
                cv2.imwrite(output_file_path1,score_map_image)
                
                filename_var = "{}-var.jpg".format(four_string)
                # 保存处理后的图像
                output_file_path2 = os.path.join(output_subfolder_path, filename_var)
                cv2.imwrite(output_file_path2,variance_map_image)
                
                filename_denoiseout = "{}-denoiseout.jpg".format(four_string)
                # 保存处理后的图像
                output_file_path3 = os.path.join(output_subfolder_path, filename_denoiseout)
                cv2.imwrite(output_file_path3,output_image)
                    
                    
                    
                    
                    
                    
                    
                    
                # # 应用转换（增加batch维度，归一化等）
                # image_tensor = transform(image).to(device)
                
                # # 去噪处理（假设模型输出也是batch x channels x height x width）
                # with torch.no_grad():
                #     denoised_image_tensor = model(image_tensor)
                #     denoised_image_tensor = denoised_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 移除batch维度，并转换回HWC格式
                
                # # 将处理后的图像转换回PIL Image格式
                # denoised_image = Image.fromarray((denoised_image_tensor * 255).astype(np.uint8))
                
                # # 保存处理后的图像
                # output_file_path = os.path.join(output_subfolder_path, filename)
                # denoised_image.save(output_file_path)

print("去噪处理完成！")





 
        
        
        
        
        
        
        
        
        # # 获取相同切片序号的所有图片  
        # image_files = os.listdir(output_qiepian)  
        # image_files.sort()  #给放到listdir里的out_folder文件排序
        # #print("----------------image_files.sort()------------",image_files)    #排序为：flair,t1,t1ce,t2
        # same_slice_images = {}  # 用字典存储相同切片序号的图像文件名列表，键为切片序号，值为文件名列表  
        # for file in image_files:  
        #     if file.endswith(".jpg"):  # 只处理jpg文件
        #         parts = file.split('-')  
        #         slice_number = int(parts[2])   # 提取切片序号 
        #         same_slice_images.setdefault(slice_number, []).append(os.path.join(output_qiepian, file))  # 将图片文件名添加到列表中  
        
    
        # # 不同切片序号的图像读取去噪  
        # for slice_number, image_files in same_slice_images.items():  
        #     images = []  
        #     for file in image_files:  
        #         image = Image.open(file).convert("L")  # 转换为灰度图，如果需要其他颜色空间请修改此处参数  
        #         images.append(np.array(image))  # 将PIL图像转换为numpy数组，以便后续处理  
              
        #     for img in images:  
    	# 		#input_image = read_img(args.input_filename)
        #         input_image =img.astype('float32')
        #         print("-------input_image---",input_image.shape)
        #         #input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).unsqueeze(0).cuda()
        #         input_var =  torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).cuda()
        #         print("-------input_var--",input_var.shape) # torch.Size([1, 1, 160, 160])

        #         with torch.no_grad():
        #             noise_level_est, output, score_map, variance_map = model(input_var)


                
        #         print("-------noise_level_est--",noise_level_est.shape) #torch.Size([1, 1, 160, 160])
        #         print("-------output--",output.shape) #torch.Size([1, 1, 160, 160])
        #         print("-------score_map--",score_map.shape) #torch.Size([1, 1, 80, 80])
        #         print("-------variance_map--",variance_map.shape) #torch.Size([1, 1, 80, 80])
        #         #可视化不确定性图和方差
        #         score_mapx = score_map  # 随机生成示例数据
        #         variance_mapx = variance_map.abs()  # 随机生成示例数据，并确保方差是非负的

                
        #         score_map_image = chw_to_hwc(score_map[0,...].cpu().numpy())
        #         score_map_image = np.uint8(np.round(np.clip(score_map_image, 0, 1) * 255.))[: ,: ,::-1]

        #         variance_map_image = chw_to_hwc(variance_map[0,...].cpu().numpy())
        #         variance_map_image = np.uint8(np.round(np.clip(variance_map_image, 0, 1) * 255.))[: ,: ,::-1]

        #         output_image = chw_to_hwc(output[0,...].cpu().numpy())
        #         output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

               
        #         deoutimg_dirr = os.path.join(pathfile,"deoutimg") 
        #         deoutimg_dir = os.path.join(deoutimg_dirr,two_string)   
        #         if not os.path.exists(deoutimg_dir):  
        #             os.makedirs(deoutimg_dir)  
                    
        #         cv2.imwrite(os.path.join(deoutimg_dir,"{}-score_map.jpg".format("%05d"%(slice_number))),score_map_image)
        #         cv2.imwrite(os.path.join(deoutimg_dir,"{}-variance_map_np.jpg".format("%05d"%(slice_number))),variance_map_image)
        #         cv2.imwrite(os.path.join(deoutimg_dir,"{}-output.jpg".format("%05d"%(slice_number))),output_image)

