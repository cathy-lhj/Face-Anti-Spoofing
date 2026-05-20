# 忽略警告信息
import warnings

warnings.filterwarnings('ignore')

# 导入OpenCV相关功能
import cv2
from cv2 import rectangle, imshow, waitKey, destroyAllWindows, CascadeClassifier

# 导入深度学习相关模块
import numpy as np
from retinex import automatedMSRCR  # 导入MSRCR图像处理函数
from attention import attention_model  # 导入论文中的注意力模型
import time

# --- 1. 初始化设置 ---
# 使用 CAP_DSHOW 在 Windows 上启动摄像头更快更稳定
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 加载人脸检测器
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# 加载模型与权重
model = attention_model(1, backbone='MobileNetV3', shape=(299, 299, 3))
model.load_weights('ver-2-weight-63-1.00-0.88-0.00179.hdf5')
print("--- 人脸反欺诈系统已启动 ---")

# 设置视频录制参数
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_final2.avi', fourcc, 20.0, (640, 480))

# --- 2. 算法参数 ---
WINDOW_SIZE = 10  # 每个人的滑动窗口长度
THRESHOLD = 0.65  # 判定阈值（若照片拦不住，请尝试调低至 0.3-0.5）

# 用于存放多张脸独立分数的字典 { face_id: [scores_list] }
face_trackers = {}

# --- 3. 主循环 ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 识别人脸
    bboxes = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # 当前帧活跃的人脸ID集合，用于清理离开画面的脸
    current_frame_face_ids = []

    for box in bboxes:
        ox, oy, ow, oh = box

        # --- A. 坐标计算与边界保护 ---
        # 向上偏移约20%以包含额头，防止截取到负数坐标
        y_start = max(0, int(oy - 0.2 * oh))
        y_end = min(frame_h, int(oy + 1.2 * oh))
        x_start = max(0, int(ox))
        x_end = min(frame_w, int(ox + 1.0 * ow))

        # --- B. 多脸独立ID分配 ---
        # 使用位置中心点作为简易ID识别（适用于非剧烈运动场景）
        face_id = f"{int(ox / 50)}_{int(oy / 50)}"
        current_frame_face_ids.append(face_id)
        if face_id not in face_trackers:
            face_trackers[face_id] = []

        # --- C. 图像预处理 ---
        face_img = frame[y_start:y_end, x_start:x_end]
        if face_img.size == 0: continue

        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_rgb, (299, 299))

        # MSRCR流处理
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        msr_img = automatedMSRCR(np.expand_dims(img_gray, -1), [10, 20, 30])
        msr_img = cv2.cvtColor(msr_img[:, :, 0], cv2.COLOR_GRAY2RGB)

        # --- D. 模型预测 ---
        input_rgb = np.expand_dims(img_resize / 255.0, 0)
        input_msr = np.expand_dims(msr_img / 255.0, 0)
        preds = model.predict([input_rgb, input_msr])
        current_score = preds[0][0]

        # --- E. 独立滑动窗口平滑 ---
        face_trackers[face_id].append(current_score)
        if len(face_trackers[face_id]) > WINDOW_SIZE:
            face_trackers[face_id].pop(0)

        avg_score = sum(face_trackers[face_id]) / len(face_trackers[face_id])

        # --- F. 判定与可视化 ---
        if avg_score > THRESHOLD:
            label = f"Fake: {avg_score:.2f}"
            color = (0, 0, 255)  # 红色-欺诈
        else:
            label = f"Real: {avg_score:.2f}"
            color = (0, 255, 0)  # 绿色-活体

        rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
        cv2.putText(frame, label, (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- G. 清理已离开画面的人脸数据 ---
    # 防止字典无限增大
    active_keys = list(face_trackers.keys())
    for k in active_keys:
        if k not in current_frame_face_ids and len(face_trackers[k]) > 0:
            face_trackers[k].pop(0)  # 逐渐清空不在画面中的脸的分数

    # 显示与保存
    out.write(frame)
    imshow('Face Anti-Spoofing Detection System', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print("系统已正常关闭。")