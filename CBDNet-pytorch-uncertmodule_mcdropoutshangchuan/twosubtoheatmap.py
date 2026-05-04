import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('G:\\MPU\\16MRI2\\yyy--CBDNetexperiment-uncertainty\\addrician00070t1_den500epoch.jpg')
img2 = cv2.imread('G:\\MPU\\16MRI2\\yyy--CBDNetexperiment-uncertainty\\openneuro_01667_t1-00070clean.jpg')

# 检查图像是否读取成功
if img1 is None or img2 is None:
    print("Error: One or both images could not be read.")
    exit()

# 检查图像尺寸是否相同
if img1.shape != img2.shape:
    print("Error: Images have different sizes.")
    exit()

# 图像相减并乘以100（假设这是为了放大差异）
img3 = cv2.subtract(img1, img2) * 20

# 将结果转换为绝对值，因为相减可能导致负值
img3 = np.absolute(img3)

# 将结果转换为8位无符号整数类型，以便应用颜色映射
img3 = np.uint8(img3)

# 应用颜色映射来创建热力图
heatmap = cv2.applyColorMap(img3, cv2.COLORMAP_JET)

# 保存结果图像
cv2.imwrite('G:\\MPU\\16MRI2\\yyy--CBDNetexperiment-uncertainty\\uncertainty5002_heatmap.jpg', heatmap)

print("Image subtraction completed and saved as uncertainty1000_heatmap.jpg")