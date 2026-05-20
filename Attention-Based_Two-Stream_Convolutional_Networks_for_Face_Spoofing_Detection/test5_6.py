import sys
import cv2
import numpy as np
import warnings
import tensorflow as tf
from keras import backend as K
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# 导入核心算法模块
from retinex import automatedMSRCR
from attention import attention_model

warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ---------------------------------------------------------
# 1. 算法推理线程
# ---------------------------------------------------------
class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, str, float, bool)

    def __init__(self):
        super().__init__()
        self.WINDOW_SIZE = 10
        self.THRESHOLD = 0.65
        self.face_trackers = {}

        self.session = K.get_session()
        self.graph = tf.compat.v1.get_default_graph()

        with self.graph.as_default():
            with self.session.as_default():
                self.model = attention_model(1, backbone='MobileNetV3', shape=(299, 299, 3))
                self.model.load_weights('ver-2-weight-63-1.00-0.88-0.00179.hdf5')
                self.model._make_predict_function()

        self.classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_h, frame_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bboxes = self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))

            current_frame_face_ids = []
            display_label, display_score, display_is_real = "SEARCHING...", 0.0, True

            for box in bboxes:
                ox, oy, ow, oh = box
                y_start, y_end = max(0, int(oy - 0.2 * oh)), min(frame_h, int(oy + 1.2 * oh))
                x_start, x_end = max(0, int(ox)), min(frame_w, int(ox + 1.0 * ow))

                face_id = f"{int(ox / 50)}_{int(oy / 50)}"
                current_frame_face_ids.append(face_id)
                if face_id not in self.face_trackers: self.face_trackers[face_id] = []

                face_img = frame[y_start:y_end, x_start:x_end]
                if face_img.size == 0: continue

                img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                img_resize = cv2.resize(img_rgb, (299, 299))
                img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
                msr_img_raw = automatedMSRCR(np.expand_dims(img_gray, -1), [10, 20, 30])
                msr_img = cv2.cvtColor(msr_img_raw[:, :, 0], cv2.COLOR_GRAY2RGB)

                with self.graph.as_default():
                    with self.session.as_default():
                        input_rgb, input_msr = np.expand_dims(img_resize / 255.0, 0), np.expand_dims(msr_img / 255.0, 0)
                        preds = self.model.predict([input_rgb, input_msr], verbose=0)
                        current_score = preds[0][0]

                self.face_trackers[face_id].append(current_score)
                if len(self.face_trackers[face_id]) > self.WINDOW_SIZE: self.face_trackers[face_id].pop(0)

                avg_score = sum(self.face_trackers[face_id]) / len(self.face_trackers[face_id])
                if avg_score > self.THRESHOLD:
                    display_label, display_is_real, color = "Fake", False, (0, 0, 255)
                else:
                    display_label, display_is_real, color = "Real", True, (0, 255, 0)

                display_score = avg_score
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)

            active_keys = list(self.face_trackers.keys())
            for k in active_keys:
                if k not in current_frame_face_ids and len(self.face_trackers[k]) > 0: self.face_trackers[k].pop(0)

            self.change_pixmap_signal.emit(frame, display_label, display_score, display_is_real)
        cap.release()


# ---------------------------------------------------------
# 2. 精简后的 UI 设计
# ---------------------------------------------------------
class AntiSpoofingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于双流注意力机制的人脸反欺诈系统")
        self.setMinimumSize(1000, 600)
        self.resize(1100, 700)

        # 深色背景
        self.setStyleSheet("background-color: #05070A; color: white;")
        self.initUI()

        self.thread = DetectionThread()
        self.thread.change_pixmap_signal.connect(self.update_screen)
        self.thread.start()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # 左侧：视频显示
        self.image_label = QLabel("正在启动摄像头...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 2px solid #1A1E23;
            border-radius: 8px;
            background-color: #000000;
        """)
        layout.addWidget(self.image_label, stretch=7)

        # 右侧：状态面板
        self.right_panel = QVBoxLayout()

        # 结果卡片
        self.info_card = QFrame()
        self.info_card.setStyleSheet("""
            QFrame {
                background-color: #0D1117;
                border: 1px solid #30363D;
                border-radius: 12px;
            }
            QLabel { border: none; background: transparent; }
        """)
        card_layout = QVBoxLayout(self.info_card)
        card_layout.setContentsMargins(30, 50, 30, 50)
        card_layout.setSpacing(15)

        # 1. 状态文字 (Real / Fake) - 颜色动态变化
        self.res_en_label = QLabel("WAITING")
        self.res_en_label.setFont(QFont('Segoe UI', 42, QFont.Bold))
        self.res_en_label.setAlignment(Qt.AlignCenter)
        self.res_en_label.setStyleSheet("color: #484F58;")
        card_layout.addWidget(self.res_en_label)

        # 2. 置信度数字 - 颜色动态变化，保持原有数字大小
        self.score_num_label = QLabel("0.00")
        self.score_num_label.setFont(QFont('Consolas', 45))
        self.score_num_label.setAlignment(Qt.AlignCenter)
        self.score_num_label.setStyleSheet("color: #484F58;")
        card_layout.addWidget(self.score_num_label)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #30363D;")
        card_layout.addWidget(line)

        # 3. 下方状态详情 - 固定白色文字
        self.detail_label = QLabel("状态：等待检测\n置信度：0%")
        self.detail_label.setFont(QFont('Microsoft YaHei', 15))
        self.detail_label.setStyleSheet("color: #FFFFFF; line-height: 160%; margin-top: 10px;")
        card_layout.addWidget(self.detail_label)

        self.right_panel.addWidget(self.info_card)
        self.right_panel.addStretch()  # 将卡片推向顶部

        layout.addLayout(self.right_panel, stretch=3)



    def update_screen(self, cv_img, label, score, is_real):
        # 1. 视频缩放渲染 (保持不变)
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        if label == "SEARCHING...":
            self.res_en_label.setText("Scan")
            self.res_en_label.setStyleSheet("color: #58A6FF;")
            self.score_num_label.setText("0.00")
            self.detail_label.setText("状态：扫描中...\n置信度：--")
        else:
            # --- 修改部分开始 ---
            # 无论 Real 还是 Fake，直接使用原始 score
            display_confidence = score
            # 将 score (0-1) 转换为百分比
            percentage = int(display_confidence * 100)

            # 更新界面文字
            self.res_en_label.setText(label)
            self.score_num_label.setText(f"{display_confidence:.4f}")  # 建议保留四位小数更具技术感

            if is_real:
                color_hex = "#3FB950"  # 绿色
                status_text = f"状态：真人\n系统得分：{display_confidence:.4f}"
            else:
                color_hex = "#F85149"  # 红色
                status_text = f"状态：攻击风险\n系统得分：{display_confidence:.4f}"
            # --- 修改部分结束 ---

            # 设置动态颜色
            self.res_en_label.setStyleSheet(f"color: {color_hex};")
            self.score_num_label.setStyleSheet(f"color: {color_hex};")
            # 下方详情文字保持白色
            self.detail_label.setText(status_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AntiSpoofingApp()
    window.show()
    sys.exit(app.exec_())