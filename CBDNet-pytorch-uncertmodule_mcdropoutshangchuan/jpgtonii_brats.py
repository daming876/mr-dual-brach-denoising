import os
import glob
import SimpleITK as sitk
import numpy as np

def convert_jpg_to_nii(input_root, output_root):
    # 遍历200个原始病例文件夹
    for case_dir in sorted(glob.glob(os.path.join(input_root, 'BraTS-GLI-*'))):
        case_id = os.path.basename(case_dir).split('-')[-2]
        output_case_dir = os.path.join(output_root, f'BraTS-GLI-{case_id}')
        os.makedirs(output_case_dir, exist_ok=True)

        # 获取所有模态类型
        all_files = glob.glob(os.path.join(case_dir, '*.jpg'))
        modalities = list(set([f.split('-')[-2] for f in all_files]))
        
        # 按模态处理图像序列
        for modality in modalities:
            # 获取当前模态的所有切片文件
            modality_files = sorted(
                [f for f in all_files if f.endswith(f'-{modality}-denoiseout.jpg')],
                key=lambda x: int(x.split('-')[-3])
            )

            # 读取并堆叠切片
            slice_arrays = []
            for img_file in modality_files:
                img = sitk.ReadImage(img_file)
                slice_arrays.append(sitk.GetArrayFromImage(img))
            
            # 创建3D NIfTI图像
            volume = np.stack(slice_arrays, axis=0)
            nii_img = sitk.GetImageFromArray(volume)
            nii_img.SetSpacing([1.0, 1.0, 1.0])  # 设置默认空间参数
            nii_img.SetOrigin([0.0, 0.0, 0.0])

            # 保存为压缩格式
            output_path = os.path.join(output_case_dir, f'{modality}.nii.gz')
            sitk.WriteImage(nii_img, output_path)
            print(f'生成文件：{output_path}')

if __name__ == "__main__":
    input_dir = "/mnt/gemlab_data_2/jiang/CBDNet/brats_denoised_nii/calibration_denoised3000epochs/"
    output_dir = "/mnt/gemlab_data_2/jiang/CBDNet/brats_denoised_nii/calibration_denoised3000epochsnii/"
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)
    convert_jpg_to_nii(input_dir, output_dir)
