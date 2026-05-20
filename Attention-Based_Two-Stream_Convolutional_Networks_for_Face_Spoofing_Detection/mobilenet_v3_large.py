#这是MobileNetV3 Large模型的Keras实现代码，是MobileNetV3的更大、更强版本
"""MobileNet v3 Large models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""
# MobileNetV3 Large版本，比Small版本更大、更准确，但计算量也更大

# 导入Keras相关模块
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape
from keras.utils.vis_utils import plot_model

from mobilenet_base import MobileNetBase
# 导入相同的基类，复用相同的构建块

# 定义MobileNetV3_Large类
class MobileNetV3_Large(MobileNetBase):
    """MobileNetV3 Large版本，更强大但参数更多的网络"""
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

        super(MobileNetV3_Large, self).__init__(shape, n_class, alpha)
        # 调用父类MobileNetBase的构造函数
        # 与Small版本使用相同的基类
        self.include_top = include_top
        # 是否包含顶部分类层的标志

    def build(self, plot=False):
        """构建MobileNetV3 Large网络

                # 参数说明
                    plot: 布尔值，是否绘制模型结构图

                # 返回值
                    model: Model对象，构建的模型
        """

        #创建输入层
        inputs = Input(shape=self.shape)
        # 输入形状如(299, 299, 3)

        # 1. 初始卷积块（与Small版本相同）
        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')
        # 16个通道，3x3卷积，步长2，h-swish激活
        # 下采样到原来的一半分辨率

        # 2. 瓶颈层序列 - Large版本有更多、更深的层
        x = self._bottleneck(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
        # 参数说明：
        #   x: 输入
        #   16: 输出通道数
        #   (3, 3): 卷积核大小
        #   e=16: 扩展因子16倍
        #   s=1: 步长1，保持分辨率
        #   squeeze=False: 不使用SE注意力
        #   nl='RE': ReLU激活

        x = self._bottleneck(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')# 输出24通道，扩展64倍，步长2下采样，ReLU激活
        x = self._bottleneck(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE') # 保持24通道，扩展72倍，步长1保持分辨率
        x = self._bottleneck(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')# 输出40通道，5x5卷积核，扩展72倍，步长2，使用SE注意力，ReLU激活

        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        # 两个40通道的瓶颈层，都使用SE注意力

        x = self._bottleneck(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS')
        # 输出80通道，3x3卷积，扩展240倍，步长2，h-swish激活
        # 从这里开始使用h-swish激活函数

        x = self._bottleneck(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        # 四个80通道的瓶颈层，都没有SE注意力

        x = self._bottleneck(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS')# 输出112通道，扩展480倍，使用SE注意力
        x = self._bottleneck(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS')# 另一个112通道的瓶颈层，扩展672倍
        x = self._bottleneck(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS') # 输出160通道，5x5卷积，扩展672倍，步长2，有SE注意力

        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        # 最后两个160通道的瓶颈层

        # 3. 最后的特征提取
        x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
        # 1x1卷积，将通道数调整到960，h-swish激活
        # Large版本输出960维特征，比Small的576维多

        x = GlobalAveragePooling2D()(x)# 全局平均池化，得到960维的特征向量

        # 以下是注释掉的分类头部
        # x = Reshape((1, 1, 960))(x)# 重塑为(1,1,960)形状
        #
        # x = Conv2D(1280, (1, 1), padding='same')(x) # 扩展到1280维
        #
        # x = self._return_activation(x, 'HS') # h-swish激活
        #
        # if self.include_top:
        #     x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x) # 分类层
        #     x = Reshape((self.n_class,))(x)# 重塑输出形状

        # 创建模型
        model = Model(inputs, x)
        # 构建Keras模型

        if plot:
            plot_model(model, to_file='images/MobileNetv3_large.png', show_shapes=True)
            # 可选：绘制并保存模型结构图
        return model

## 测试代码
# model = MobileNetV3_Large((299,299,3), 1).build()
# print(model.summary())# 可以取消注释来测试模型构建
