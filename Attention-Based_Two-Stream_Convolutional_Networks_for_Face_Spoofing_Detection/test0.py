#忽略警告信息
import warnings
warnings.filterwarnings('ignore')
# 导入OpenCV相关功能
from cv2 import imread, imshow, waitKey, destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import cv2
# 导入深度学习相关模块
from keras.models import load_model
from retinex import automatedMSRCR# 导入MSRCR图像处理函数
from attention import attention_model# 导入论文中的注意力模型
import numpy as np
import time # 时间模块

# 打开摄像头
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)
# 加载人脸检测器和训练好的模
classifier = CascadeClassifier('haarcascade_frontalface_default.xml') # Haar级联分类器，用于人脸检测
model = attention_model(1, backbone='MobileNetV3', shape=(299, 299, 3))# 创建注意力双流模型
model.load_weights('ver-2-weight-63-1.00-0.88-0.00179.hdf5')# 加载预训练权重
#model.load_weights('weight-65-1.00-1.00-0.00004.hdf5')# 加载预训练权重
print(model.summary())# 打印模型结构

# 设置视频录制参数
fourcc = cv2.VideoWriter_fourcc(*'MJPG')#视频编码格式
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))#创建VideoWriter对象，20fps，640x480分辨率

#开始主循环
while True:
    ret, frame = cap.read()#读取一帧摄像头画面
    # ret: 是否成功读取，frame: 图像帧
    bboxes = classifier.detectMultiScale(frame)# 检测当前帧中的人脸，返回边界框列表

    for box in bboxes:# 遍历每个检测到的人脸
        x, y, width, height = box # 获取边界框坐标
        x2, y2 = int(x + 1.0*width), int(y + 1.2*height)# 调整右下角坐标，高度增加20%
        x, y = int(x-0.0*width), int(y-0.2*height)# 调整左上角坐标，向上移动20%
        # 提取和预处理人脸区域
        img = frame[y:y2, x:x2]# 从原图中截取人脸区域
        print(type(img))# 调试：打印图像类型
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# 从BGR转为RGB格式
        img = cv2.resize(img, (299, 299))# 调整大小为模型输入尺寸
        #生成MSRCR图像
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)# RGB转为灰度图
        new_img = np.expand_dims(new_img, -1)# 增加一维：(299,299) -> (299,299,1)
        new_img = automatedMSRCR(new_img, [10, 20, 30])# 应用MSRCR处理，参数为多尺度[10,20,30]
        new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)# 从灰度转回RGB

        #模型预测
        preds = model.predict([np.expand_dims(img / 255.0, 0), # 标准化RGB图像并增加批次维度
                               np.expand_dims(new_img / 255.0, 0)]) # 标准化MSRCR图像并增加批次维度
        # 根据预测结果绘制边界框
        if preds[0][0] > 0.90:# 预测值大于0.9认为是虚假人脸
            rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)# 红色框(0,0,255)
        else:
            rectangle(frame, (x, y), (x2, y2), (0,255,0), 1)# 绿色框(0,255,0)
        out.write(frame)# 将当前帧写入输出视频
    #imshow('face detection', frame) # 显示处理后的图像


    if cv2.waitKey(25) & 0xFF == ord('q'):# 等待25毫秒，如果按'q'键则退出
        break
#释放资源
cap.release()# 释放摄像头
out.release()# 释放VideoWriter
cv2.destroyAllWindows()# 关闭所有OpenCV窗口