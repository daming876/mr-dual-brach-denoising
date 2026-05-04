#把jpg文件重新命名，直接python renamecleanjpg.py就可以执行

import os   
import shutil 
def rename_jpg_files(old_directory,new_directory):
    # 获取目录下的所有文件和子目录
    for filename in os.listdir(old_directory):
        # 检查文件是否以.jpg结尾
        if filename.lower().endswith('.jpg'):
            # 构建旧的文件路径
            old_file_path = os.path.join(old_directory, filename)
            # 构建新的文件名和路径
            new_filename = f"GT_SRGB_{filename}"
            new_file_path = os.path.join(new_directory, new_filename)
            # 重命名并复制文件
            shutil.copy2(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")

# 指定要处理的文件夹路径
old_directory_path = '/mnt/gemlab_data_2/jiang/HWformer/data/images/train/openneuro-cleanrun01/'
new_directory_path='/mnt/gemlab_data_2/jiang/HWformer/data/images/train/openneuro-clean-noisy-sigmahun/openneuro-clean-noisy-sigmahunhe/'
# 调用函数进行重命名
rename_jpg_files(old_directory_path,new_directory_path)