#这是MobileNetV3 Small模型的Keras实现代码，用于论文中的双流注意力网络的主干网络。
"""MobileNet v3 small models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""
# 这是论文《Searching for MobileNetV3》的Keras实现
# MobileNetV3是轻量高效的CNN架构，适合移动端和边缘计算

# 导入Keras相关模块
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape
from keras.utils.vis_utils import plot_model

from mobilenet_base import MobileNetBase
# 导入基础类，包含MobileNet通用的构建块


# 定义MobileNetV3_Small类
class MobileNetV3_Small(MobileNetBase):
    """MobileNetV3 Small版本，更轻量的网络"""

    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        """初始化函数

                # 参数说明
                    input_shape: 整数或3个整数的元组/列表，输入张量的形状
                    n_class: 整数，类别数量
                    alpha: 整数，宽度乘子（控制网络宽度）
                    include_top: 布尔值，是否包含分类层

                # 返回值
                    MobileNetv3模型
                """

        super(MobileNetV3_Small, self).__init__(shape, n_class, alpha)
        # 调用父类MobileNetBase的构造函数
        # 传递shape, n_class, alpha参数

        self.include_top = include_top
        # 是否包含顶部分类层的标志



    def build(self, plot=False):
        """构建MobileNetV3 Small网络

              # 参数说明
                  plot: 布尔值，是否绘制模型结构图

              # 返回值
                  model: Model对象，构建的模型
              """

        # 创建输入层
        inputs = Input(shape=self.shape)
        # Input是Keras的输入层
        # self.shape是从父类继承的输入形状，如(299, 299, 3)


        # 以下是MobileNetV3 Small的架构构建：

        # 1. 初始卷积块
        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')
        # 参数说明：
        #   inputs: 输入张量
        #   16: 输出通道数
        #   (3, 3): 卷积核大小
        #   strides=(2, 2): 步长为2，下采样一半
        #   nl='HS': 激活函数为h-swish
        # 这是一个标准的卷积块，进行初步特征提取

        # 2. 瓶颈层序列（深度可分离卷积）
        x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        # 参数说明：
        #   x: 输入
        #   16: 输出通道数
        #   (3, 3): 卷积核大小
        #   e=16: 扩展因子（expansion factor）
        #   s=2: 步长，下采样
        #   squeeze=True: 使用SE（Squeeze-and-Excitation）注意力模块
        #   nl='RE': 激活函数为ReLU
        # 这是MobileNet的核心构建块：逆残差瓶颈层
        x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')# 输出24通道，扩展因子72，步长2，无SE注意力，ReLU激活
        x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE') # 输出24通道，扩展因子88，步长1（保持分辨率），无SE注意力
        x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS')# 输出40通道，5x5卷积核，扩展因子96，步长2，有SE注意力，h-swish激活

        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        # 两个相同的瓶颈层，保持分辨率

        x = self._bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS')
        # 两个48通道的瓶颈层

        x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS')
        # 输出96通道，步长2，下采样

        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        # 最后两个瓶颈层

        # 3. 最后的特征提取
        x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        # 1x1卷积，将通道数调整到576，h-swish激活
        # 这是最后的特征映射层

        x = GlobalAveragePooling2D()(x)
        # 全局平均池化，将(H,W,C)的特征图池化为(1,1,C)
        # 对空间维度进行平均，得到576维的特征向量


        # 以下是注释掉的分类头部（分类任务使用）
        # x = Reshape((1, 1, 576))(x) # 将特征向量重塑为(1,1,576)的形状
        #
        # x = Conv2D(1280, (1, 1), padding='same')(x)# 1x1卷积，将576维特征映射到1280维
        # x = self._return_activation(x, 'HS') # 应用h-swish激活函数
        #
        # if self.include_top:
        #     x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
        #     # 分类层，输出n_class个类别的概率
        #     x = Reshape((self.n_class,))(x)
        #     # 重塑为(n_class,)的形状

        # 创建模型
        model = Model(inputs, x)
        # 使用Keras的Model类，指定输入和输出

        # 可选：绘制模型结构图
        if plot:
            plot_model(model, to_file='images/MobileNetv3_small.png', show_shapes=True)
            # 将模型结构图保存为PNG文件，显示各层形状

        return model
        # 返回构建的模型
