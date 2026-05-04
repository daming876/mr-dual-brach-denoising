import os  
import shutil  
  
# 假设imagesTr是包含所有MR图像对的文件夹路径  
images_dir = '/mnt/gemlab_data/Medical_image_database/MRI/segmentation/MR_Dataset/8ProstateX/imagesTr/'  
images_new = '/mnt/gemlab_data_2/jiang/CBDNet/prostate_origin_8ProstateX_files/'   
# 遍历images_dir中的所有文件  
for filename in os.listdir(images_dir):  
    # 检查文件名是否以'ProstateX-'开头，并且是.nii.gz文件  
    if filename.startswith('ProstateX-') and filename.endswith('.nii.gz'):  
        # 提取文件名中的前两个字段（即文件夹名）  
        # 假设文件名格式为 'ProstateX-XXXX-...'  
        folder_name = '-'.join(filename.split('-')[:2])  
          
        # 构造完整的新文件夹路径  
        new_folder_path = os.path.join(images_new, folder_name)  
          
        # 如果文件夹不存在，则创建它  
        if not os.path.exists(new_folder_path):  
            os.makedirs(new_folder_path)  
          
        # 构造原始文件的完整路径  
        old_file_path = os.path.join(images_dir, filename)  
          
        # 检查是否已经有对应的另一个文件（adc或t2w）在新文件夹中  
        # 如果没有，则直接移动；如果有，则假设已经配对，不需要再次移动  
        # 这里我们假设文件名除了后缀外其他部分完全相同  
        base_filename = filename.rsplit('.', 1)[0]  # 去掉.nii.gz后缀  
        other_file = f"{base_filename.replace('-adc', '-t2w')}.gz" if '-adc' in filename else f"{base_filename.replace('-t2w', '-adc')}.gz"  
        other_file_path = os.path.join(images_dir, other_file)  
          
        # 如果另一个文件也存在，则一起移动  
        if os.path.exists(other_file_path):  
            # 移动文件到新文件夹  
            shutil.copy(old_file_path, new_folder_path)  
            shutil.copy(other_file_path, new_folder_path)  
            print(f"Moved {filename} and {other_file} to {new_folder_path}")  
        else:  
            # 如果另一个文件不存在，可以打印一个警告或者跳过，这里选择打印警告  
            print(f"Warning: {other_file} does not exist, skipping {filename}")  
  
print("All files have been moved to their respective folders.")