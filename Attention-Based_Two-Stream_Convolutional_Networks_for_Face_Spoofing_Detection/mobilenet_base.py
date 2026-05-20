#这是MobileNetV3的基础构建模块代码，包含了MobileNetV3架构的核心组件实现。
"""MobileNet v3 models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""
# 引用MobileNetV3论文，这是实现MobileNetV3架构的基础类

# 导入Keras层
from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape

from keras import backend as K
# 导入构建MobileNetV3所需的所有Keras层

# 定义MobileNetBase基类
class MobileNetBase:
    """MobileNetV3的基础类，包含所有共享的构建块"""

    def __init__(self, shape, n_class, alpha=1.0):
        """初始化基础类

                # 参数说明
                    input_shape: 整数或3个整数的元组/列表，输入张量的形状
                    n_class: 整数，类别数量
                    alpha: 整数，宽度乘子（控制网络宽度）
                """

        self.shape = shape # 存储输入形状
        self.n_class = n_class # 存储类别数，2 分类，0 = 真人脸，1 = 假人脸
        self.alpha = alpha # 存储宽度乘数，控制模型大小

    # ReLU6激活函数
    def _relu6(self, x):
        """ReLU6激活函数，将最大值限制在6.0"""
        """Relu 6
        """
        return K.relu(x, max_value=6.0)
    # 这是MobileNet系列的标准激活函数，限制最大值为6有助于数值稳定性

    # Hard-Swish激活函数
    def _hard_swish(self, x):
        """Hard-swish激活函数，Swish的高效近似"""
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0
    # 公式：x * ReLU6(x + 3) / 6
    # 这是MobileNetV3引入的新激活函数，比Swish计算更高效

    # 激活函数选择器
    def _return_activation(self, x, nl):
        """激活函数选择器

               # 参数说明
                   x: 张量，卷积层的输入张量
                   nl: 字符串，非线性激活类型

               # 返回值
                   输出张量
               """
        if nl == 'HS':# Hard-Swish
            x = Activation(self._hard_swish)(x)
        if nl == 'RE':# ReLU6
            x = Activation(self._relu6)(x)

        return x

    #标准卷积块
    def _conv_block(self, inputs, filters, kernel, strides, nl):
        """卷积块：包含卷积、批归一化和激活函数

                # 参数说明
                    inputs: 张量，卷积层的输入张量
                    filters: 整数，输出空间的维度（通道数），特征图数量
                    kernel: 整数或2个整数的元组/列表，卷积核的宽和高
                    strides: 整数或2个整数的元组/列表，卷积的步长
                    nl: 字符串，非线性激活类型

                # 返回值
                    输出张量
                """

        # 确定通道轴的位置（不同后端格式不同）
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        # 1. 标准卷积层
        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        # padding='same': 输出尺寸与输入相同（考虑步长）
        # strides: 卷积步长
        # 2. 批归一化
        x = BatchNormalization(axis=channel_axis)(x) # 沿通道轴进行批归一化，让数据分布更稳定，模型训练更快、更不容易崩！

        # 3. 激活函数
        return self._return_activation(x, nl)
    """它就是一个标准卷积打包块：
卷积 → 归一化 → 激活"""

    # SE注意力模块
    """这是 SE（Squeeze-and-Excitation）通道注意力模块 —— 你论文里最大的亮点！
    作用：让模型自动 “聪明地” 关注重要特征，忽略无用特征，大幅提高人脸防伪准确率！"""
    def _squeeze(self, inputs):
        """Squeeze-and-Excitation（压缩和激励）注意力模块

               # 参数说明
                   inputs: 张量，卷积层的输入张量
               """

        input_channels = int(inputs.shape[-1])# 获取输入通道数,拿到特征图有多少个通道（比如 32、64）

        # 1. Squeeze: 全局平均池化
        x = GlobalAveragePooling2D()(inputs)# 得到每个通道的平均值
        # 2. Excitation: 两个全连接层学习通道权重
        x = Dense(input_channels, activation='relu')(x) # 第一个FC，ReLU激活
        x = Dense(input_channels, activation='hard_sigmoid')(x)# 第二个FC，hard-sigmoid激活
        # 3. 重塑形状以匹配输入
        x = Reshape((1, 1, input_channels))(x)# 重塑为(1,1,C)形状
        # 4. Scale: 通道加权
        x = Multiply()([inputs, x]) # 输入与学习到的权重逐通道相乘

        return x
    # SE模块让网络学习每个通道的重要性，增强重要特征，抑制不重要特征

    # 逆残差瓶颈块（核心构建块）
    def _bottleneck(self, inputs, filters, kernel, e, s, squeeze, nl):
        """逆残差瓶颈块 - MobileNet的核心结构

                # 参数说明
                    inputs: 张量，输入张量
                    filters: 整数，输出空间的维度
                    kernel: 整数或元组，卷积核大小
                    e: 整数，扩展因子（expansion factor）
                    s: 整数或元组，步长
                    squeeze: 布尔值，是否使用SE注意力
                    nl: 字符串，激活函数类型

                # 返回值
                    输出张量
                """
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        # 确定通道轴
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)# 获取输入形状

        # 计算扩展通道数和输出通道数
        tchannel = int(e)# 扩展后的通道数
        cchannel = int(self.alpha * filters) # 输出通道数（考虑宽度乘子alpha）

        # 判断是否使用残差连接（当步长为1且输入输出通道数相同时）
        r = s == 1 and input_shape[3] == filters

        # 1. 扩展（Pointwise卷积）：1x1卷积扩展通道
        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1), nl) # 将低维特征映射到高维空间

        # 2. 深度可分离卷积
        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        # DepthwiseConv2D: 深度卷积，每个通道独立卷积
        # depth_multiplier=1: 每个输入通道产生1个输出通道
        # 3. 批归一化和激活
        x = BatchNormalization(axis=channel_axis)(x)
        x = self._return_activation(x, nl)

        # 4. 可选：SE注意力模块
        if squeeze:
            x = self._squeeze(x)# 应用SE注意力

        # 5. 投影（Pointwise卷积）：1x1卷积压缩通道
        x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        # 将高维特征映射回低维

        # 6. 残差连接（如果条件满足）
        if r:
            x = Add()([x, inputs])# 残差连接

        return x
    # 这是MobileNetV3的核心：扩展→深度卷积→压缩的逆残差结构

    # 抽象方法
    def build(self):
        """构建模型的方法，由子类实现"""
        pass
