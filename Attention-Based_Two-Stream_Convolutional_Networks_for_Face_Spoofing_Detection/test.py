# 忽略警告信息
import warnings

warnings.filterwarnings('ignore')

# 导入OpenCV相关功能
import cv2
from cv2 import rectangle, imshow, waitKey, destroyAllWindows, CascadeClassifier

# 导入深度学习相关模块
from keras.models import load_model
from retinex import automatedMSRCR  # 导入MSRCR图像处理函数
from attention import attention_model  # 导入论文中的注意力模型
import numpy as np
import time

# --- 1. 初始化设置 ---
# 打开摄像头，CAP_DSHOW 在 Windows 上更稳定
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 加载人脸检测器
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# 加载模型与权重
model = attention_model(1, backbone='MobileNetV3', shape=(299, 299, 3))
model.load_weights('ver-2-weight-63-1.00-0.88-0.00179.hdf5')
print("--- 模型加载成功 ---")
# print(model.summary())

# 设置视频录制参数
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 换成 XVID 兼容性更好
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# --- 2. 算法参数 ---
score_window = []
WINDOW_SIZE = 10  # 判定窗口长度
THRESHOLD = 0.65  # 判定阈值（可根据实验调整）

# --- 3. 主循环 ---
print("系统运行中，按 'q' 退出...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 警告：无法获取画面")
        continue

    # 获取当前帧尺寸，用于边界保护
    frame_h, frame_w = frame.shape[:2]

    # 灰度化加速检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bboxes = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for box in bboxes:
        ox, oy, ow, oh = box

        # --- 核心改进：带边界保护的坐标计算 ---
        # 向上偏移20% (y)，高度增加20%
        y_start = max(0, int(oy - 0.2 * oh))
        y_end = min(frame_h, int(oy + 1.2 * oh))
        x_start = max(0, int(ox))
        x_end = min(frame_w, int(ox + 1.0 * ow))

        # 提取人脸区域
        img = frame[y_start:y_end, x_start:x_end]

        # 检查图像是否为空（防止 cvtColor 报错）
        if img is None or img.size == 0:
            continue

        # --- 图像预处理 ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_rgb, (299, 299))

        # 生成MSRCR流图像
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        msr_img = automatedMSRCR(np.expand_dims(img_gray, -1), [10, 20, 30])
        msr_img = cv2.cvtColor(msr_img[:, :, 0], cv2.COLOR_GRAY2RGB)

        # --- 模型预测 ---
        input_rgb = np.expand_dims(img_resize / 255.0, 0)
        input_msr = np.expand_dims(msr_img / 255.0, 0)
        preds = model.predict([input_rgb, input_msr])

        current_score = preds[0][0]  # 假设输出是欺诈概率


        # --- 4. 滑动窗口多帧融合算法 ---
        score_window.append(current_score)
        if len(score_window) > WINDOW_SIZE:
            score_window.pop(0)

        avg_score = sum(score_window) / len(score_window)

        # --- 5. 结果显示 ---
        if avg_score > THRESHOLD:
            # 红色表示欺诈
            label = f"Fake: {avg_score:.2f}"
            color = (0, 0, 255)
        else:
            # 绿色表示活体
            label = f"Real: {avg_score:.2f}"
            color = (0, 255, 0)

        # 在原图画框和文字
        rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
        cv2.putText(frame, label, (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 录制与实时显示
    out.write(frame)
    imshow('Face Anti-Spoofing Detection', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# --- 4. 释放资源 ---
print("正在关闭系统...")
cap.release()
out.release()
cv2.destroyAllWindows()