#Retinex图像增强算法，主要用于生成MSRCR和MSRCP图像
import numpy as np
import cv2

#单尺度Retinex函数
def singleScaleRetinex(img, sigma):
    """单尺度Retinex算法，是Retinex的基础单元"""
    img[img==0] = 1# 避免对0取对数，将0值替换为1
    # 核心公式：Retinex = log(原始图像) - log(模糊图像)
    retinex = np.log10(img) - np.expand_dims(np.log10(cv2.GaussianBlur(img, (0, 0), sigma)),-1)
    return retinex
    # 解释：sigma是高斯核的标准差，控制模糊程度
    # 高斯模糊模拟光照变化，log(原始)-log(模糊)得到反射分量

#多尺度Retinex函数
def multiScaleRetinex(img, sigma_list):
    """多尺度Retinex，综合多个尺度的Retinex结果"""
    retinex = np.zeros_like(img, dtype='float')# 创建与img相同形状的全零数组
    for sigma in sigma_list: # 遍历每个尺度
        retinex += singleScaleRetinex(img, sigma)# 累加每个尺度的Retinex结果
    retinex = retinex / len(sigma_list)# 取平均值得到最终结果

    # 以下是注释掉的标准化代码（可选步骤）：这些是图像归一化，把增强后的值变回 0~255 的正常图像范围。
    # retinex = retinex + np.amin(retinex)# 使最小值为0
    # retinex = retinex / np.amax(retinex)# 归一化到[0,1]
    # retinex = retinex * 255.0# 缩放到[0,255]
    # retinex = retinex.astype('uint8')# 转为8位无符号整数

    return retinex
# 解释：sigma_list是不同尺度的列表，如[10, 20, 30]
# 多尺度组合能更好地处理不同频率的图像信息

# 颜色恢复函数,给 MSR 增强后的图像恢复颜色，不让图像变灰、变偏色，让色彩更自然、更鲜艳。
def colorRestoration(img, alpha, beta):
    """颜色恢复，补偿Retinex处理后的颜色损失"""
    img_sum = np.sum(img, axis=2, keepdims=True)# 计算每个像素的RGB通道和
    # 颜色恢复公式：beta * (log(alpha*img) - log(img_sum))
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration
# 解释：alpha是增益参数，beta是控制颜色恢复强度的参数
# 用于恢复Retinex处理后可能损失的颜色信息
"""由于多尺度 Retinex（MSR）增强后会出现颜色失真与暗淡问题，
本文引入颜色恢复模块对增强后图像进行色彩补偿。该模块通过对数运
算重构各颜色通道的相对关系，有效恢复图像的真实色彩信息，使 MSR 
处理后的图像在光照均匀化的同时保持色彩自然与真实。"""

# 简单颜色平衡函数,图像增强最后一步：对比度拉伸 + 颜色平衡。把图像的对比度拉开，让图像更清晰、不发灰、不过亮 / 过暗。
def simplestColorBalance(img, low_clip, high_clip):
    """简单的颜色平衡，通过裁剪直方图两端来增强对比度"""
    total = img.shape[0] * img.shape[1]# 计算总像素数
    for i in range(img.shape[2]):# 对每个颜色通道分别处理
        unique, counts = np.unique(img[:, :, i], return_counts=True)# 获取像素值和频数
        current = 0
        for u, c in zip(unique, counts):# 遍历像素值
            if float(current) / total < low_clip:# 找到low_clip分位点的值
                low_val = u
            if float(current) / total < high_clip:# 找到high_clip分位点的值
                high_val = u
            current += c
        # 将像素值限制在[low_val, high_val]范围内
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img
# 解释：low_clip和high_clip是裁剪比例，如0.01和0.99
# 去掉直方图两端1%的极端值，增强中间部分的对比度

# MSRCR主函数,这是完整的 MSRCR 算法（带颜色恢复的多尺度 Retinex），
# 作用就是：输入一张原图 → 输出光照均匀、颜色自然、对比度清晰的 MSR 增强图。
def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0# 转换为float64并加1避免0值
    # 三步处理：
    img_retinex = multiScaleRetinex(img, sigma_list)# 1. 多尺度Retinex
    img_color = colorRestoration(img, alpha, beta)# 2. 颜色恢复
    img_msrcr = G * (img_retinex * img_color + b)# 3. 线性组合
    # 下面对每个通道进行标准化到[0,255]
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))# 限制范围并转uint8
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip) # 4. 颜色平衡

    return img_msrcr
# 解释：这是论文中使用的MSR算法的完整实现
# G和b是增益和偏置参数，用于调整输出强度

# 自动MSRCR函数这是一个 “不用手动调参” 的自动版 MSRCR。
# 它自动分析图像像素分布，自动裁剪异常值，自动标准化，输出增强好的图像。
def automatedMSRCR(img, sigma_list):
    """自动化的MSRCR，自动确定像素值范围，无需手动调参"""
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)# 多尺度Retinex

    for i in range(img_retinex.shape[2]):# 对每个通道单独处理
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)  # 统计像素值分布
        for u, c in zip(unique, count): # 找到0值像素的数量
            if u == 0:
                zero_count = c
                break
        # 自动确定裁剪范围
        low_val = unique[0] / 100.0 # 最小值
        high_val = unique[-1] / 100.0# 最大值
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1: # 如果负值像素少于0值的10%
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1: # 如果正值像素少于0值的10%
                high_val = u / 100.0
                break
        # 应用裁剪
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        # 标准化到[0,255]
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)# 转为8位图像

    return img_retinex
# 解释：这是代码中实际调用的函数
# 自动确定像素值裁剪范围，更鲁棒

# MSRCP函数，MSRCP = 只增强亮度、不破坏颜色的 Retinex 算法。
# 它先对亮度做增强，再按原始色彩比例恢复颜色，所以颜色最自然、最不偏色。
def MSRCP(img, sigma_list, low_clip, high_clip):
    """MSRCP（带颜色保留的多尺度Retinex），保持颜色更自然"""
    img = np.float64(img) + 1.0
    # 计算强度通道（RGB平均值）
    intensity = np.sum(img, axis=2) / img.shape[2]
    # 对强度通道应用Retinex
    retinex = multiScaleRetinex(intensity, sigma_list)
    # 增加维度以便计算
    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)
    # 颜色平衡
    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)
    # 标准化强度
    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0
    # 重建彩色图像
    img_msrcp = np.zeros_like(img)

    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])# 原图最大通道值
            # 计算缩放因子，保持颜色比例
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            # 应用缩放
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)# 减1并转uint8

    return img_msrcp
# 解释：与MSRCR不同，MSRCP先处理强度通道，再恢复颜色
# 能更好地保持原始颜色，避免颜色失真

# 示例代码
# img = cv2.imread("fake/0000.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#把颜色通道从 BGR → RGB，保证颜色正常
# img = cv2.resize(img, (299, 299))#统一图像大小，方便后续做人脸防伪299*299

# 测试和可视化代码
if __name__ == '__main__':
    # 测试图像列表
    images = ['fake/0000.png', 'fake/0.png', 'fake/0001.png', 'fake/0001_00_00_01_0.jpg',
              'real/0000.png', 'real/0.png', 'real/0001.png', 'real/0001_00_00_01_0.jpg']

    import matplotlib.pyplot as plt

    # 创建4x4的图像网格
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4

    for index, img in enumerate(images):# 遍历每张测试图像
        img = cv2.imread(img)# 读取图像
        img = cv2.resize(img, (299, 299))# 调整大小

        # 显示原始图像
        fig.add_subplot(4, 4, 2 * (index) + 1)# 左侧位置
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# BGR转RGB
        plt.imshow(tmp)

        # 处理并显示MSRCR结果
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转为灰度
        img = np.expand_dims(img, -1)# 增加通道维度
        new_img = automatedMSRCR(img, [2, 4, 8])# 应用MSRCR
        # 可选：使用不同的Retinex函数
        # new_img = multiScaleRetinex(img, [5, 10, 15])
        # print(new_img.shape)
        # new_img = cv2.cvtColor(new_img[:,:,0], cv2.COLOR_GRAY2RGB)  #把灰度增强图 → 转成 RGB 彩色图

        # 显示处理结果
        fig.add_subplot(4, 4, 2 * (index) + 2)# 右侧位置
        plt.imshow(new_img[:,:,0], cmap='gray') # 显示单通道灰度图
    plt.show()# 显示所有图像

# 额外的测试代码这
# img = multiScaleRetinex(img, [10, 20, 30])
# img = automatedMSRCR(img, [10,20,30])
# print(img)
# print(img.shape)
# cv2.imshow("img", img)#弹出窗口，显示处理后的图片
# cv2.waitKey(0)#等待按键关闭窗口