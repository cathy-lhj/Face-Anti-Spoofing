# 忽略Python运行时弹出的警告信息，让控制台更干净
import warnings
warnings.filterwarnings('ignore')

from cv2 import imread, imshow, waitKey, destroyAllWindows# 从OpenCV库导入需要的功能：读图、显示图、等待按键、关闭窗口
from cv2 import CascadeClassifier# 导入人脸检测分类器
from cv2 import rectangle# 导入画矩形框函数
import cv2# 导入完整OpenCV库
from keras.models import load_model# 从Keras导入模型加载函数（深度学习）
from retinex import automatedMSRCR# 导入MSRCR图像增强函数（论文核心：光照预处理）
from attention import attention_model# 导入自定义的双流注意力模型（论文核心网络）
import numpy as np
import time
# cap = cv2.VideoCapture(0)# 打开电脑摄像头（0=默认摄像头）
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


classifier = CascadeClassifier('haarcascade_frontalface_default.xml')# 加载Haar级联分类器，用于在画面中找到人脸位置
model = attention_model(1, backbone='MobileNetV3', shape=(299, 299, 3))# 创建双流注意力模型：2分类，主干网络MobileNetV3，输入尺寸299x299
model.load_weights('ver-2-weight-63-1.00-0.88-0.00179.hdf5')# 加载训练好的模型权重文件
print(model.summary())# 打印模型结构（层数、参数数量）


fourcc = cv2.VideoWriter_fourcc(*'MJPG')# 设置视频编码格式：MJPG
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))# 创建视频保存对象：保存为output.avi，帧率20，分辨率640x480

while True:# 开始无限循环，实时读取摄像头画面
    # 读取一帧图像
    # ret：是否读取成功
    # frame：当前帧图像
    ret, frame = cap.read()

    # ===================== 关键：读不到也不退出！=====================
    if not ret:
        print("⚠️  警告：读取画面失败，但继续运行...")
        continue  # 不退出！

    # 新增：获取画面尺寸，用于校验人脸框坐标
    h, w = frame.shape[:2]
    bboxes = classifier.detectMultiScale(frame) # 用人脸检测器检测当前帧中的所有人脸

    for box in bboxes: # 遍历每一张检测到的人脸
        try:
            x, y, width, height = box # 取出人脸框的坐标：x、y、宽度、高度
            x2, y2 = int(x + 1.0*width), int(y + 1.2*height) # 调整人脸框右下角坐标，高度扩大20%
            x, y = int(x-0.0*width), int(y-0.2*height)# 调整左上角坐标，向上移动20%

            # 新增：校验裁剪区域是否有效（避免空数组）
            if x >= x2 or y >= y2:
                continue  # 无效区域跳过


            img = frame[y:y2, x:x2]  # 从原图中裁剪出人脸区域
            print(type(img))  # 打印图像类型（调试用）
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 将图像从BGR格式转为RGB格式（模型需要RGB）
            img = cv2.resize(img, (299, 299))# 将图像缩放到模型输入尺寸 299x299

            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 生成MSRCR增强图像：先转灰度图
            new_img = np.expand_dims(new_img, -1) # 增加一个维度，变成 (299,299,1)
            new_img = automatedMSRCR(new_img, [10, 20, 30]) # 使用MSRCR进行图像增强（多尺度）
            new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)# 将增强后的图像转回RGB

            # 模型预测：输入两张图（RGB图 + MSRCR增强图）
            preds = model.predict([np.expand_dims(img / 255.0, 0), np.expand_dims(new_img / 255.0, 0)])

            # 如果预测值>0.9，判定为【假人脸】，画红色框
            if preds[0][0] > 0.90:
                rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)
            # 否则判定为【真人脸】，画绿色框
            else:
                rectangle(frame, (x, y), (x2, y2), (0,255,0), 1)
            out.write(frame) # 将当前帧写入视频文件
        except Exception as e:  # 捕获所有异常并打印
            print(f"❌ 处理人脸时出错：{e}")
            continue  # 出错后跳过当前人脸，不终止程序
        imshow('face detection', frame) # 显示实时检测画面


        if cv2.waitKey(60) & 0xFF == ord('q'): # 等待25毫秒，如果按下 q 键，退出循环
            break

cap.release()# 释放摄像头
out.release()# 释放视频保存器
cv2.destroyAllWindows()# 关闭所有OpenCV窗口

