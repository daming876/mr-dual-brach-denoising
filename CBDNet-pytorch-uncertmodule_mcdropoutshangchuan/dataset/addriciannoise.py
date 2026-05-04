#读取文件夹 filesa （干净图像）中的灰度 JPG 图片，给它们添加 Rician 噪声，并将结果以灰度 JPG 格式保存到文件夹 filesb（保存为sigma_img） 中。
#直接执行python addriciannoise.py就可以
import os
import numpy as np
import cv2

def generate_rician_noise(size, signal_strength):
    # 生成实部和虚部信号
    real_part = np.random.normal(0, 1, size)
    imag_part = np.random.normal(0, 1, size)

    amplitude = np.sqrt(real_part**2 + imag_part**2)  # 计算幅值图像

    amplitude *= signal_strength / np.mean(amplitude)  # 根据信号强度调整噪声分布

    rician_noise = amplitude * np.exp(1j * np.random.uniform(0, 2 * np.pi, size))  # 生成Rician噪声

    #return rician_noise.real.astype(np.float32)  # 只返回实部作为噪声，并保持为float32类型
    return rician_noise.astype(np.complex64)  # 返回复数的Rician噪声 

def add_rician_noise_to_image(image, noise_strength):
    noise = generate_rician_noise(image.shape, noise_strength)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)  # 确保值在0-255之间
    return noisy_image

def process_images(input_folder, output_folder, noise_strength_range):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图片

            # 从噪声强度范围中随机选择一个噪声强度
            noise_strength = np.random.uniform(*noise_strength_range)

            # 添加Rician噪声
            noisy_image = add_rician_noise_to_image(image, noise_strength)

            # 构造新文件名
            new_filename = f"SIGMA_SRGB_{filename}"
 
            # 保存加噪后的图片
            output_path = os.path.join(output_folder, new_filename)
            print(f"add rician noise:{output_path}")
            cv2.imwrite(output_path, noisy_image)

# 使用示例
input_folder = '/mnt/gemlab_data_2/jiang/HWformer/data/images/train/openneuro-cleanrun01/'
output_folder = '/mnt/gemlab_data_2/jiang/HWformer/data/images/train/openneuro-clean-noisy-sigmahun/openneuro-clean-noisy-sigmahunhe/'
noise_strength_range = (10, 100)  # 你可以根据需要调整这个范围

process_images(input_folder, output_folder, noise_strength_range)

