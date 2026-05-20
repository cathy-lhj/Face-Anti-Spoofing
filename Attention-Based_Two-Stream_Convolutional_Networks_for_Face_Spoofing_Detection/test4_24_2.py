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
# 1. 算法推理线程（负责计算、标注和录制）
# ---------------------------------------------------------
class DetectionThread(QThread):
    # 信号只传回处理后的图像（包含框选和文字）
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.WINDOW_SIZE = 10
        self.THRESHOLD = 0.65
        self.face_trackers = {}
        self.running = True

        # 初始化模型
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

                # 在视频帧上绘制 (BGR颜色)
                color = (0, 255, 0) if is_real else (0, 0, 255)
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
                cv2.putText(frame, f"{label_text}: {avg_score:.4f}", (x_start, y_start - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)
            self.change_pixmap_signal.emit(frame)

        cap.release()
        out.release()


# ---------------------------------------------------------
# 2. UI 界面类
# ---------------------------------------------------------
class AntiSpoofingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸反欺诈系统 - 双流注意力监测")
        self.resize(1100, 700)
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
        layout.setSpacing(25)

        # --- 左侧：监控显示 (占据大部分空间) ---
        self.image_label = QLabel("正在启动系统...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet("border: 2px solid #30363D; border-radius: 12px; background-color: #000;")
        layout.addWidget(self.image_label, stretch=8)

        # --- 右侧：说明面板 ---
        right_panel = QVBoxLayout()

        table_title = QLabel("💡 系统置信度评分说明")
        table_title.setFont(QFont('Microsoft YaHei', 12, QFont.Bold))
        table_title.setStyleSheet("color: #8B949E; margin-bottom: 10px; border: none;")
        right_panel.addWidget(table_title)

        # 创建并配置表格
        self.table = QTableWidget(3, 2)
        self.table.setHorizontalHeaderLabels(["分数区间", "判定类型"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setStyleSheet("""
            QTableWidget { 
                background-color: #0D1117; 
                gridline-color: #30363D; 
                border: 1px solid #30363D; 
                border-radius: 10px; 
                font-size: 10pt;
            }
            QHeaderView::section { 
                background-color: #161B22; 
                color: #8B949E; 
                padding: 8px; 
                border: 1px solid #30363D; 
                font-weight: bold;
            }
            QTableWidget::item { 
                color: #C9D1D9; 
                padding: 10px; 
            }
        """)

        data = [
            ("0.00 ~ 0.40", "Real (活体)"),
            ("0.40 ~ 0.65", "Uncertain"),
            ("0.65 ~ 1.00", "Fake (欺诈)")
        ]
        for i, (c1, c2) in enumerate(data):
            self.table.setItem(i, 0, QTableWidgetItem(c1))
            self.table.setItem(i, 1, QTableWidgetItem(c2))


        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedHeight(300)  # 适当调整表格高度

        right_panel.addWidget(self.table)

        # 底部添加一些系统参数说明（可选）
        tech_note = QLabel(
            "\n[系统运行状态]\n模型: MobileNetV3-Attention\n输入尺寸: 299x299\n平滑策略: 滑动窗口 (N=10)")
        tech_note.setStyleSheet("color: #484F58; font-family: 'Consolas'; font-size: 10pt;")
        right_panel.addWidget(tech_note)

        right_panel.addStretch()
        layout.addLayout(right_panel, stretch=3)

    def update_screen(self, cv_img):
        # 转换 OpenCV 图像为 Qt 图片
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)

        # 按比例缩放并显示，不撑开布局
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AntiSpoofingApp()
    window.show()
    sys.exit(app.exec_())