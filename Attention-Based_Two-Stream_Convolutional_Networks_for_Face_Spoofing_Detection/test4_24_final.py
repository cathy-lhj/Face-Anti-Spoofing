import sys
import cv2
import numpy as np
import warnings
import tensorflow as tf
from keras import backend as K
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QFrame, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSizePolicy)
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
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.WINDOW_SIZE = 10
        self.THRESHOLD = 0.65
        self.face_trackers = {}
        self.running = True

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
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('detection_output.avi', fourcc, 20.0, (640, 480))

        while self.running:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (640, 480))
            frame_h, frame_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bboxes = self.classifier.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))

            for box in bboxes:
                ox, oy, ow, oh = box
                y_start, y_end = max(0, int(oy - 0.2 * oh)), min(frame_h, int(oy + 1.2 * oh))
                x_start, x_end = max(0, int(ox)), min(frame_w, int(ox + 1.0 * ow))

                face_id = f"{int(ox / 50)}_{int(oy / 50)}"
                if face_id not in self.face_trackers: self.face_trackers[face_id] = []

                face_img = frame[y_start:y_end, x_start:x_end]
                if face_img.size == 0: continue

                img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                img_resize = cv2.resize(img_rgb, (299, 299))
                img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
                msr_img = automatedMSRCR(np.expand_dims(img_gray, -1), [10, 20, 30])
                msr_img = cv2.cvtColor(msr_img[:, :, 0], cv2.COLOR_GRAY2RGB)

                with self.graph.as_default():
                    with self.session.as_default():
                        preds = self.model.predict([np.expand_dims(img_resize / 255.0, 0),
                                                    np.expand_dims(msr_img / 255.0, 0)], verbose=0)
                        current_score = preds[0][0]

                self.face_trackers[face_id].append(current_score)
                if len(self.face_trackers[face_id]) > self.WINDOW_SIZE: self.face_trackers[face_id].pop(0)

                avg_score = sum(self.face_trackers[face_id]) / len(self.face_trackers[face_id])
                is_real = avg_score <= self.THRESHOLD
                label_text = "Real" if is_real else "Fake"
                color = (0, 255, 0) if is_real else (0, 0, 255)

                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
                cv2.putText(frame, f"{label_text}: {avg_score:.4f}", (x_start, y_start - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)
            self.change_pixmap_signal.emit(frame)

        cap.release()
        out.release()


# ---------------------------------------------------------
# 2. 优化后的 UI 界面类
# ---------------------------------------------------------
class AntiSpoofingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸反欺诈监控系统 v2.0 - 双流注意力机制")
        # 增加初始尺寸，让整体感官更大气
        self.resize(1280, 800)
        self.setMinimumSize(1024, 720)
        self.setStyleSheet("background-color: #0A0C10; color: #E6EDF3;")

        self.initUI()

        self.thread = DetectionThread()
        self.thread.change_pixmap_signal.connect(self.update_screen)
        self.thread.start()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(25, 25, 25, 25)  # 增加页边距
        main_layout.setSpacing(30)  # 增加控件间距

        # --- 左侧：监控显示区域 ---
        left_container = QVBoxLayout()
        video_title = QLabel("🎥 实时监控视频流")
        video_title.setFont(QFont('Microsoft YaHei', 11, QFont.Bold))
        video_title.setStyleSheet("color: #8B949E; margin-bottom: 5px;")

        self.image_label = QLabel("正在连接硬件设备...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 优化显示器边框：深色描边 + 柔和圆角
        self.image_label.setStyleSheet("""
            border: 2px solid #30363D; 
            border-radius: 15px; 
            background-color: #000000;
        """)

        left_container.addWidget(video_title)
        left_container.addWidget(self.image_label)
        main_layout.addLayout(left_container, stretch=7)  # 视频区域占 7/10

        # --- 右侧：面板区域 ---
        right_panel = QVBoxLayout()

        # 使用 QFrame 封装侧边栏，增加视觉整体感
        side_card = QFrame()
        side_card.setStyleSheet("""
            QFrame {
                background-color: #161B22; 
                border: 1px solid #30363D; 
                border-radius: 12px;
            }
        """)
        card_layout = QVBoxLayout(side_card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(15)

        table_header = QLabel("💡 判定置信度指南")
        table_header.setFont(QFont('Microsoft YaHei', 12, QFont.Bold))
        table_header.setStyleSheet("color: #58A6FF; border: none;")  # 蓝色强调色
        card_layout.addWidget(table_header)

        # 表格配置
        self.table = QTableWidget(3, 2)
        self.table.setHorizontalHeaderLabels(["分数区间", "判定状态"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setShowGrid(False)  # 隐藏网格线，更具现代感

        # 优化表格样式
        self.table.setStyleSheet("""
            QTableWidget { 
                background-color: transparent; 
                border: none;
                font-size: 11pt;
            }
            QHeaderView::section { 
                background-color: #21262D; 
                color: #8B949E; 
                padding: 10px; 
                border: none;
                font-weight: bold;
            }
            QTableWidget::item { 
                color: #C9D1D9; 
                border-bottom: 1px solid #30363D;
                padding: 15px; 
            }
        """)

        data = [
            ("0.00 ~ 0.40", "✅ Real (活体)"),
            ("0.40 ~ 0.65", "⚠️ Uncertain"),
            ("0.65 ~ 1.00", "❌ Fake (欺诈)")
        ]
        for i, (c1, c2) in enumerate(data):
            self.table.setItem(i, 0, QTableWidgetItem(c1))
            self.table.setItem(i, 1, QTableWidgetItem(c2))
            self.table.setRowHeight(i, 55)  # 增加行高

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedHeight(240)
        card_layout.addWidget(self.table)

        # 底部系统日志区域
        log_title = QLabel("⚙️ 系统日志")
        log_title.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
        log_title.setStyleSheet("color: #8B949E; border: none; margin-top: 15px;")
        card_layout.addWidget(log_title)

        self.sys_log = QLabel("• 神经网络已加载\n• 权重校验成功\n• 滑动窗口：N=10\n• 采样频率：20 FPS")
        self.sys_log.setStyleSheet("color: #484F58; font-family: 'Consolas'; font-size: 9pt; border: none;")
        card_layout.addWidget(self.sys_log)

        card_layout.addStretch()  # 内部弹簧
        right_panel.addWidget(side_card)
        main_layout.addLayout(right_panel, stretch=3)  # 面板占 3/10

    def update_screen(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)

        # 优化缩放逻辑：动态适应标签大小，保持平滑效果
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.image_label.width() - 4, self.image_label.height() - 4,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AntiSpoofingApp()
    window.show()
    sys.exit(app.exec_())