#忽略所有警告信息，让输出更清晰
import warnings
warnings.filterwarnings('ignore')

#导入自定义模块和第三方库
from attention import attention_model  #导入注意力模型（论文核心）
from datagen import DataGenerator   #导入数据生成器
from keras_radam import RAdam   #RAdam优化器
from keras_lookahead import Lookahead   # Lookahead优化器包装
#from keras.utils import multi_gpu_model    # 多GPU训练（但代码中未实际使用）
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau    # 训练回调函数

#导入标准库
from sklearn.model_selection import train_test_split    # 数据分割（但代码中未使用）
import os     # 文件路径操作
import cv2    # OpenCV（但代码中未使用）
from tqdm import tqdm    # 进度条显示
import matplotlib.pyplot as plt   # 可视化
import argparse     # 命令行参数解析，让Python脚本可以从命令行接收参数，无需修改代码就能调整配置。

parser = argparse.ArgumentParser()
parser.add_argument("--bs", help="batch size", required=True)
parser.add_argument("--dim", help="dim", required=True)
parser.add_argument("--backbone", help="backbone architecture", required=True)
args = parser.parse_args()
"""# 使用不同的配置
python train.py --bs 64 --dim 256 --backbone vgg16
python train.py --bs 32 --dim 224 --backbone resnet50
# 使用EfficientNet
python train.py --bs 128 --dim 380 --backbone efficientnetb0
# 查看帮助信息
python train.py --help"""

bs = int(args.bs)#将参数转换为合适的类型
dim = (int(args.dim),int(args.dim))# # 图像尺寸转元组，如(224,224)


#准备训练集图像路径
train_images = []
#for folder in ['train\\fake', 'train\\real']:# 遍历fake和real文件夹
for folder in ['CASIA\\train\\fake', 'CASIA\\train\\real']:
	for image in os.listdir(folder):
		train_images.append(os.path.join(folder, image))

#为训练图像创建标签（进度条显示）
train_labels = {}  # 字典：{图像路径: 标签}
for image in tqdm(train_images):   # tqdm显示进度条
    if image.split('\\')[1] == 'real': # 路径分割，如'train\\real\\001.jpg'
        train_labels[image] = 0#真
    else:
        train_labels[image] = 1#假

# 准备验证集图像路径（同上）
val_images = []
#for folder in ['test\\fake', 'test\\real']:
for folder in ['CASIA\\test\\fake', 'CASIA\\test\\real']:
	for image in os.listdir(folder):
		val_images.append(os.path.join(folder, image))

val_labels = {}
for image in tqdm(val_images):
    if image.split('\\')[1] == 'real':
        val_labels[image] = 0
    else:
        val_labels[image] = 1

#创建训练和验证数据生成器
train_gen = DataGenerator(train_images, train_labels, batch_size=bs, dim=dim, type_gen='train')
val_gen = DataGenerator(val_images, val_labels, batch_size=bs, dim=dim, type_gen='test')
#测试数据生成器，显示批数据形状
X, Y = train_gen[0]
print(len(X), X[0].shape, X[1].shape)# 应为2, 分别显示RGB流和MSR流的形状
print(Y)# 显示对应的标签


#可视化第一个批次的RGB图像
fig = plt.figure(figsize=(8, 8))
columns = 4
rows = bs//columns# 计算行数

for i in range(1, columns * rows + 1):
    fig.add_subplot(rows, columns, i)# 创建子图
    plt.imshow(X[0][i - 1])# 显示RGB流中的第i-1张图
plt.show()


# 可视化第一个批次的MSR图像
fig = plt.figure(figsize=(8, 8))
for i in range(1, columns * rows + 1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(X[1][i - 1]) # 显示MSR流中的第i-1张图
plt.show()

#创建注意力模型
model = attention_model(1, backbone=args.backbone, shape=(dim[0], dim[1], 3))
# 参数：1=二分类，backbone=主干网络，shape=(高,宽,通道)
print(model.summary())# 打印模型结构摘要
#配置优化器
optimizer = Lookahead(RAdam())# 使用RAdam+Lookahead优化器组合# Lookahead提升收敛稳定性，RAdam是Adam的改进版
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])# 编译模型，使用二分类交叉熵损失
#配置训练回调函数
validate_freq = 1 # 每1个epoch验证一次
start_epoch = 0# 起始epoch
# 权重文件保存路径，包含epoch、精度、验证精度、验证损失
filepath = "save_weight/"+"weight-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}-{val_loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, period=validate_freq)#保存权重参数
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.95, patience=2, verbose=1, mode='auto')# ReduceLROnPlateau：当验证损失不再下降时降低学习率
# factor=0.95: 新学习率=旧学习率×0.95
# patience=2: 连续2个epoch验证损失不下降则降低学习率

callbacks_list = [checkpoint, reduce_lr]#回调函数列表

# Train model on dataset，开始训练模型
print("FITTING")
model.fit_generator(generator=train_gen,#训练数据生成器
                    validation_data=val_gen,# 验证数据生成器
                    epochs=90,# 训练90个epoch
                    verbose=1,# 显示详细训练进度
                    callbacks=callbacks_list,# 使用定义的回调函数
                    initial_epoch=start_epoch, # 起始epoch
                    validation_freq=validate_freq, # 验证频率
                    max_queue_size=20, # 生成器队列最大大小
                    workers = 8,# 使用8个工作进程加载数据
                    )