"""人脸防伪任务中使用的双流数据生成器的完整实现代码。
这个数据生成器专门为论文中提出的"基于注意力机制的双流卷积神经网络"
（Two-Stream CNN with Attention Mechanism）设计"""
#你论文里【双流 CNN 人脸防伪】的核心前置部分：数据生成器 + 图像增强。
#导入基础库
import numpy as np
import keras
import cv2
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from retinex import automatedMSRCR
# 注释：这是论文中使用的MSRCR图像增强算法，用于生成光照不变的特征图像

# 自定义图像增强变换，对 RGB 三个颜色通道，分别、独立、随机地调整亮度和对比度
from albumentations.augmentations.functional import brightness_contrast_adjust
import albumentations as A

class IndependentRandomBrightnessContrast(A.ImageOnlyTransform):
    """独立通道亮度对比度调整 - 每个通道独立调整亮度和对比度"""
    """ Change brightness & contrast independently per channels """
    #初始化参数
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5):#亮度变化范围 ±20%，对比度，，，0.5概率执行增强
        super(IndependentRandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = A.to_tuple(brightness_limit)# 亮度调整范围
        self.contrast_limit = A.to_tuple(contrast_limit) # 对比度调整范围

    def apply(self, img, **params):
        img = img.copy()# 创建副本
        for ch in range(img.shape[2]):# 对每个颜色通道R,G,B独立处理
            # 为每个通道随机生成对比度和亮度参数
            alpha = 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1])# 对比度因子随机
            beta = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]) # 亮度偏移随机
            img[..., ch] = brightness_contrast_adjust(img[..., ch], alpha, beta)#对当前这个通道单独调整亮度和对比度
        return img
# 这个自定义变换模拟了不同光照条件下的人脸图像变化，三个通道完全独立随机变化！
"""为提升模型对色彩失真攻击的感知能力，本文设计一种独立通道亮度对比度增强策略。与传统全局亮度对比度调整不同，
该方法对 RGB 三通道分别进行独立随机的亮度与对比度调节，强化模型对不同颜色通道特征差异的学习能力，有效提升对打印、
回放等色彩失真攻击的检测性能，增强模型的泛化性与鲁棒性。"""

# Albumentations数据增强流水线
"""这是你论文里最核心、最强的「人脸防伪专用数据增强流水线」！
把几何、光照、颜色、细节、翻转全部组合在一起，
全自动给模型制造无限种真实场景，让模型训练后超强、超稳、超准！"""
albu_tfms =  A.Compose([
    # 第一阶段：几何变换（三选一）
    A.OneOf([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, value=0),# 平移缩放旋转
        A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15, # 光学畸变
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0),
        A.NoOp()# 不操作
    ]),

    # 第二阶段：亮度对比度调整（四选一）
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.5,
                                   contrast_limit=0.4),# 标准亮度对比度
        IndependentRandomBrightnessContrast(brightness_limit=0.25,
                                                        contrast_limit=0.24),# 独立通道调整
        A.RandomGamma(gamma_limit=(50, 150)), # Gamma校正
        A.NoOp()
    ]),

    # 第三阶段：颜色变换（四选一）
    A.OneOf([
        A.FancyPCA(),# PCA颜色增强
        A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),# RGB通道偏移
        A.HueSaturationValue(hue_shift_limit=5,
                             sat_shift_limit=5),# 色调饱和度调整
        A.NoOp() # 不操作
    ]),

    # 第四阶段：对比度增强（二选一）
    A.OneOf([
        A.CLAHE(),# 自适应直方图均衡化
        A.NoOp()# 不操作
    ]),

    # 第五阶段：水平翻转
    A.HorizontalFlip(p=0.5),# 50%概率水平翻转
])
# 注释：这个复杂的增强流水线模拟了真实世界中的人脸变化，提高模型泛化能力

# 定义DataGenerator类这是专门给「双流 CNN 人脸防伪模型」喂数据的工具它会同时生成两路图像：
# RGB 原图流
# MSRCR 增强图像流
# 然后交给模型训练。
class DataGenerator(keras.utils.Sequence):
    """自定义数据生成器，继承自Keras的Sequence类"""
    #'Initialization' '初始化'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32),shuffle=True, type_gen='train'):
        self.dim = dim # 图像尺寸，如(299, 299)
        self.batch_size = batch_size # 一次喂给模型多少张图
        self.labels = labels # 图像对应的标签：真假
        self.list_IDs = list_IDs # 图像路径列表
        self.shuffle = shuffle# 每个 epoch 是否打乱数据
        self.type_gen = type_gen  # 生成器类型：'train'/'test'/'predict'
        self.aug_gen = ImageDataGenerator() # Keras内置数据增强生成器
        print("all:", len(self.list_IDs), " batch per epoch", int(np.floor(len(self.list_IDs) / self.batch_size)))
        self.on_epoch_end()# 初始打乱

    # 实现__len__方法， '返回每个epoch的批次数''告诉模型1 轮训练要跑多少批数据。'
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # 实现on_epoch_end方法,'每个epoch结束时更新索引,每训练完一轮，就打乱所有图片顺序防止模型记住图片顺序，导致过拟合。'
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))# 创建索引数组
        if self.shuffle == True:
            np.random.shuffle(self.indexes)# 打乱索引

    # 实现__getitem__方法,    '生成一个批次的数据,Keras 模型需要数据时，会调用这个函数拿一个批次的数据。'
    def __getitem__(self, index):
        # 获取当前批次的索引
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 获取对应的图像路径
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # 生成数据
        X, y = self.__data_generation(list_IDs_temp)
        if self.type_gen == 'predict':# 预测模式只返回X
            return X
        else:# 训练/验证模式返回X和y
            return X, y

    # sequence_augment方法,Keras自带的数据增强
    def sequence_augment(self, img):
        '序列化数据增强 - 随机选择2-4种增强方法'
        name_list = ['rotate', 'width_shift', 'height_shift',
                     'brightness', 'flip_horizontal', 'width_zoom',
                     'height_zoom']# 增强方法名称列表,旋转。。。。。。
        dictkey_list = ['theta', 'ty', 'tx',
                        'brightness', 'flip_horizontal', 'zy',
                        'zx']# 对应Keras的参数名
        random_aug = np.random.randint(2, 5)  # 随机选择2-4种增强方法
        pick_idx = np.random.choice(len(dictkey_list), random_aug, replace=False)  # 随机选择索引

        dict_input = {} # 参数字典
        for i in pick_idx:#给选中的增强生成随机参数
            if dictkey_list[i] == 'theta': # 旋转
                dict_input['theta'] = np.random.randint(-10, 10)# -10到10度

            elif dictkey_list[i] == 'ty':  # 宽度平移
                dict_input['ty'] = np.random.randint(-20, 20)# -20到20像素

            elif dictkey_list[i] == 'tx':  # 高度平移
                dict_input['tx'] = np.random.randint(-20, 20)# -20到20像素

            elif dictkey_list[i] == 'brightness':# 亮度调整
                dict_input['brightness'] = np.random.uniform(0.75, 1.25) # 0.75-1.25倍

            elif dictkey_list[i] == 'flip_horizontal':# 水平翻转
                dict_input['flip_horizontal'] = bool(random.getrandbits(1))# 随机布尔值

            elif dictkey_list[i] == 'zy':  # 宽度缩放
                dict_input['zy'] = np.random.uniform(0.75, 1.25)# 0.75-1.25倍

            elif dictkey_list[i] == 'zx':  # 高度缩放
                dict_input['zx'] = np.random.uniform(0.75, 1.25)# 0.75-1.25倍
        img = self.aug_gen.apply_transform(img, dict_input)# 应用增强
        return img

    # albu_aug方法，'使用Albumentations进行数据增强'，albu_tfms
    def albu_aug(self, image, tfms = albu_tfms):
        seed = random.randint(0, 99999) # 随机种子

        random.seed(seed)# 设置随机种子
        image = tfms(image=image.astype('uint8'))['image']# 应用变换
        return image

    # __data_generation方法（核心双流生成）给双流 CNN 模型，同时生成两路输入：RGB 原图 + MSRCR 增强图，用来判断真假人脸
    def __data_generation(self, list_IDs_temp):
        '生成包含batch_size个样本的数据'
        # 初始化两路输入
        X = [np.empty((self.batch_size, self.dim[0], self.dim[1], 3)), # RGB流
             np.empty((self.batch_size, self.dim[0], self.dim[1], 3))]# MSR流
        Y = np.empty((self.batch_size), dtype=int) # 标签

        for i, ID in enumerate(list_IDs_temp):  # ID = 图片的路径（如 data/real/001.jpg），i = 这是批次里的第几张图
            # 1. 加载和预处理RGB图像
            img = cv2.imread(ID)# 读取图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# BGR转RGB
            img = cv2.resize(img, (self.dim[1], self.dim[0]))# 调整大小

            if self.type_gen =='train':# 训练模式
                # 对RGB图像应用数据增强
                img = self.sequence_augment(img)

                # 生成autoMSRCR增强图像
                new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)# 转灰度
                new_img = np.expand_dims(new_img, -1)# 增加通道维度
                new_img = automatedMSRCR(new_img, [10, 20, 30])# 应用MSRCR增强
                new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)# 转回RGB

                # 归一化到[0,1]两路数据归一化后喂给模型
                X[0][i] = img/255.0# RGB流
                X[1][i] = new_img/255.0# MSR流
            else: # 验证/测试模式
                # 生成MSR图像（无数据增强）
                new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                new_img = np.expand_dims(new_img, -1)
                new_img = automatedMSRCR(new_img, [10, 20, 30])
                new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)

                # 归一化
                X[0][i] = img/255.0
                X[1][i] = new_img/255.0

            Y[i] = self.labels[ID]# 获取标签，0 = 真人，1 = 假人脸（照片 / 回放）

        return X, Y   #X = [RGB 流，MSR 流]，Y = 真假标签

    # def __data_generation(self, list_IDs_temp):
    #     X = [np.empty((self.batch_size, self.dim[0], self.dim[1], 3)),  # RGB流
    #          np.empty((self.batch_size, self.dim[0], self.dim[1], 3))]  # MSR流
    #     Y = np.empty((self.batch_size), dtype=int)
    #
    #     for i, ID in enumerate(list_IDs_temp):
    #         img = cv2.imread(ID)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         img = cv2.resize(img, (self.dim[1], self.dim[0]))
    #
    #         # --- 修改后的逻辑 ---
    #         # 无论 train 还是 test，都要生成 MSR 图像
    #         temp_for_msr = img.copy()
    #
    #         # 如果是训练模式，可以额外加一些随机增强（可选）
    #         if self.type_gen == 'train':
    #             img = self.sequence_augment(img)
    #             temp_for_msr = img.copy()
    #
    #         # 生成 autoMSRCR 增强图像
    #         new_img = cv2.cvtColor(temp_for_msr, cv2.COLOR_RGB2GRAY)
    #         new_img = np.expand_dims(new_img, -1)
    #         new_img = automatedMSRCR(new_img, [10, 20, 30])
    #         new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)
    #
    #         # 统一赋值
    #         X[0][i] = img / 255.0
    #         X[1][i] = new_img / 255.0
    #         Y[i] = self.labels[ID]
    #         # ---------------------
    #     return X, Y