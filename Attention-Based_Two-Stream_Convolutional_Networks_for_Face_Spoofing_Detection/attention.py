import warnings
warnings.filterwarnings("ignore")

# 导入Keras相关模块
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dot, \
    Add, Lambda, Layer, Multiply, Concatenate, Softmax, Flatten
from keras.applications.xception import Xception
import keras.backend as K

from mobilenet_v3_large import MobileNetV3_Large
from mobilenet_v3_small import MobileNetV3_Small
# 注释：导入论文中可能使用的不同骨干网络



# 定义自定义注意力层
class Attention(Layer):
    """自定义注意力层，实现论文中的注意力融合机制"""

    def __init__(self, size, **kwargs):
        """初始化注意力层

              参数:
                  size: 特征向量的维度大小
              """
        self.trainable=True
        self.size = size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """构建层，创建可训练参数

               参数:
                   input_shape: 输入张量的形状
               """
        # print(input_shape)  # 调试用
        # 创建可训练的注意力权重向量q
        self.q = self.add_weight(name='q',
                                 shape=(1, self.size), # 形状：(1, 特征维度)
                                 initializer='ones',# 初始化为全1
                                 trainable=self.trainable)# 可训练
        super(Attention, self).build(input_shape)

    def call(self, x):
        """前向传播逻辑

               参数:
                   x: 输入列表，包含两个特征流 [stream1, stream2]

               返回:
                   加权融合后的特征
               """
        # 1. 解包两个输入流
        stream1, stream2 = x[0], x[1]

        # 2. 计算每个流的注意力得分
        # 将特征向量与注意力权重q做点积，得到注意力得分
        d1 = Lambda(lambda x: K.sum(x * self.q, axis=1, keepdims=True))(stream1) # sum over second axis# 沿axis=1求和
        d2 = Lambda(lambda x: K.sum(x * self.q, axis=1, keepdims=True))(stream2)
        # 3. 连接两个注意力得分
        ds = Concatenate(axis=1)([d1, d2])

        # d1, d2形状: (batch_size, 1)
        # ds 形状: (batch_size, 2)

        # 4. 通过Softmax计算归一化权重
        # tmp = Softmax(axis=0)(ds)
        tmp = Softmax(axis=1)(ds)
        # 调试打印语句
        # print(tmp._keras_shape)# 打印张量的Keras形状

        # 5. 提取权重
        w1 = Lambda(lambda x: x[:, 0])(tmp)# 取第一个权重
        w2 = Lambda(lambda x: x[:, 1])(tmp) # 这里应该是x[:, 1]，但代码中写成了x[:, 0]！！！！！！！！！！！！！！！！
        # 这看起来像是bug，w1和w2获取了相同的值

        # 6. 扩展维度以便广播
        w1 = Lambda(lambda x: K.expand_dims(x, -1))(w1)# 形状: (batch_size, 1)
        w2 = Lambda(lambda x: K.expand_dims(x, -1))(w2)# 形状: (batch_size, 1)

        #调试代码
        # print(w1._keras_shape)
        # print(w1.shape)
        # print(w2.shape)

        # 7. 应用注意力权重
        stream1 = Lambda(lambda x: x[0]*x[1])([stream1, w1]) # stream1 * w1
        stream2 = Lambda(lambda x: x[0]*x[1])([stream2, w2])# stream2 * w2
        # 8. 加权融合
        # result = Lambda(lambda x: x[0]+x[1])([stream1, stream2])
        result = Add()([stream1, stream2])# 两个加权后的特征相加
        # print(result.shape)#调试代码
        return result

    def compute_output_shape(self, input_shape):
        """计算输出形状

               返回与第一个输入流相同的形状
               """
        return input_shape[0]


# 74-89行：测试代码（注释状态）
# stream1 = Input(batch_shape= (2, 10*10*2048,))
# stream2 = Input(batch_shape = (2,10*10*2048,))
# shapes = 2048*100
# stream1 = Input((shapes,))
# stream2 = Input((shapes,))
# output = Attention(size=shapes)([stream1, stream2])
# print(output._keras_shape)
# model = Model(inputs=[stream1, stream2], outputs=output)
# print(model.summary())


# 构建完整的注意力模型
def attention_model(classes, backbone, shape):
    """构建完整的注意力双流模型

       参数:
           classes: 输出类别数（1表示二分类）
           backbone: 骨干网络类型 'Xception'/'MobileNetV3_Large'/'MobileNetV3_Small'
           shape: 输入图像形状

       返回:
           完整的Keras模型
       """
    # 1. 选择并构建两个相同的骨干网络
    if backbone == 'Xception':# 使用Xception作为特征提取器
        stream1 = Xception(include_top=False, weights='imagenet', input_shape=shape)
        stream2 = Xception(include_top=False, weights='imagenet', input_shape=shape)
    elif backbone == 'MobileNetV3_Large': # 使用MobileNetV3 Large
        stream1 = MobileNetV3_Large(shape, classes).build()
        stream2 = MobileNetV3_Large(shape, classes).build()
    else: # MobileNetV3_Small
        stream1 = MobileNetV3_Small(shape, classes).build()
        stream2 = MobileNetV3_Small(shape, classes).build()
        """Large 输出 960 维向量,Small 输出 576 维向量"""
    # 2. 为两个流命名以便区分
    stream1.name = 'stream1'
    stream2.name = 'stream2'
    # 3. 创建两个输入层
    input1 = Input(shape)# RGB流输入
    input2 = Input(shape)  # MSR流输入
    # 4. 前向传播
    output1 = stream1(input1) # RGB特征
    output2 = stream2(input2)# MSR特征
    # 5. 对Xception输出添加全局平均池化
    if backbone == 'Xception':
        output1 = GlobalAveragePooling2D(name='avg_pool_1')(output1)
        output2 = GlobalAveragePooling2D(name='avg_pool_2')(output2)


    # 6. 应用注意力融合
    # stream1 = Flatten()(stream1)# 如果需要展平
    # stream2 = Flatten()(stream2)
    # print(stream1.shape)
    output = Attention(size=output1.shape[1])([output1, output2])
    # print(output.shape)

    # 7. 添加分类头
    if classes==1:# 二分类问题ss
        output = Dense(classes, activation='sigmoid', name='predictions')(output)
    else: # 多分类问题
        output = Dense(classes, activation='softmax', name='predictions')(output)
    # print(output.shape)

    # 8. 构建并返回完整模型
    return Model(inputs=[input1, input2], outputs=output)

# 测试代码（注释状态）
# model = attention_model(2)
# print(model.summary())
# from keras.utils import plot_model
# plot_model(model, 'model.png', show_shapes=True)

