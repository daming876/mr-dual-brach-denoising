import cv2
import numpy as np
img1 = cv2.imread('G:\\MPU\\16MRI2\\yyy--CBDNet(cvpr2019)\\experiment\\calibration\\brats_segment_3000epochs\\BraTS-GLI-00005-000-denoisedimg&var\\BraTS-GLI-00005-000-00096-t2w-score.jpg')

#图像相减并乘以100（假设这是为了放大差异）
img3= img1*10

#将结果转换为绝对值，因为相减可能导致负值
img3 = np.absolute(img3)

#将结果转换为8位无符号整数类型，以便应用颜色映射
img3 = np.uint8(img3)

#应用颜色映射来创建热力图
heatmap = cv2.applyColorMap(img3, cv2.COLORMAP_JET)

#交换红色和蓝色通道，使红色变为蓝色
heatmap = heatmap[:, :, [2, 1, 0]]

#保存结果图像
cv2.imwrite('G:\\MPU\\16MRI2\\yyy--CBDNet(cvpr2019)\\experiment\\calibration\\brats_segment_3000epochs\\BraTS-GLI-00005-000-denoisedimg&var\\BraTS-GLI-00005-000-00096-t2w-scoreheat.jpg', heatmap)

print("Image subtraction completed and saved as uncertainty1000_heatmap.jpg")