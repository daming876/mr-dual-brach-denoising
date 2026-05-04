#保存jpg图片为nii
# 模态列表  
#modalities = ['t1', 't2w', 'flair', 'adc']  #模态齐全用这个
pathfile="/mnt/gemlab_data/Medical_image_database/MRI/segmentation/MR_Dataset/prostate_denoised_nii/"        
     

deoutimg_dirr = os.path.join(pathfile,"deoutimg") 
deoutimg_dir = os.path.join(deoutimg_dirr,two_string)   

if not os.path.exists(deoutimg_dir):  
    os.makedirs(deoutimg_dir) 

modalities = ['adc','t2w']  #模态不齐全时，只写出、保存已有模态
# 存储每个模态的切片数据  
data_dict = {modality: [] for modality in modalities}  
deoutimg_files=os.listdir(deoutimg_dir)
deoutimg_files.sort()
# 遍历deoutimg_dir中的文件  
for filename in deoutimg_files:  
if filename.endswith('.jpg'):  
    # 提取模态和切片号
        
    slice_number, _, modalitys  = filename.split('-')[:3]  
    print("-----------slice_number-------",slice_number) 
    print("-----------modalitys-------",modalitys) 
    modality = modalitys.split('.')[0]
    print("-----------modality-------",modality) 
    
    # 确保模态是预期的之一  
    if modality in modalities:  
        # 读取图像  
        img_path = os.path.join(deoutimg_dir, filename)  
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        
        # 如果是该模态的第一个图像，则记录图像尺寸  
        if not data_dict[modality]:  
            image_height, image_width = img.shape  
            print(f"Detected {modality} image dimensions: {image_height}x{image_width}")  
        
        # 将图像添加到对应模态的数据列表中  
        data_dict[modality].append(img)  

# 将每个模态的数据保存为NIfTI文件  .键被赋值给 modality，值被赋值给 slices
for modality, slices in data_dict.items(): 
print(f"-----modality：{modality} and ---slices：{len(slices)}")   
if slices:  
    image_3d = sitk.GetImageFromArray(np.stack(slices, axis=0).astype(np.float32))          
    # 设置图像属性（如间距和方向），这里假设所有切片都是等间距的且方向为RAS+  
    # 假设切片间距为1mm，并且方向为默认的RAS+  
    image_3d.SetSpacing((1.0, 1.0, 1.0))  # X, Y, Z间距  
    image_3d.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # RAS+ 方向  
    
    # 保存NIfTI文件
    deoutimg_nii = os.path.join(pathfile,"nii")
    
    #保存到不同的文件夹下用下面代码
    # deoutimg_savenii = os.path.join(deoutimg_nii,two_string)   
    # if not os.path.exists(deoutimg_savenii):  
    #     os.makedirs(deoutimg_savenii)    
    # nii_file_path = os.path.join(deoutimg_savenii, f"{two_string}-{modality}.nii")  
    # sitk.WriteImage(image_3d, nii_file_path)  
    
    #保存到同一文件夹下
    if not os.path.exists(deoutimg_nii):  
        os.makedirs(deoutimg_nii)    
    nii_file_path = os.path.join(deoutimg_nii, f"{two_string}-{modality}.nii")  
    sitk.WriteImage(image_3d, nii_file_path)  
    
    print(f"Saved {modality} NIfTI file to {nii_file_path}")  