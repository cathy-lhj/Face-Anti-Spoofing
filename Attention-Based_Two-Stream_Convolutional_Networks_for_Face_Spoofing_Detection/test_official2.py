import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from attention import attention_model
from retinex import automatedMSRCR

# ====================== 配置项 ======================
TEST_DIR = "CASIA/test"
# MODEL_WEIGHT = "weight-22-0.99-1.00-0.00000.hdf5" # 最优权重
MODEL_WEIGHT = "ver-2-weight-63-1.00-0.88-0.00179.hdf5"
INPUT_SIZE = (299, 299)
BACKBONE = "MobileNetV3"
# ====================================================

# 加载模型
model = attention_model(1, backbone=BACKBONE, shape=(299, 299, 3))
model.load_weights(MODEL_WEIGHT)
print("✅ 模型加载完成\n")

# 获取测试集路径和标签
def get_test_paths():
    real_dir = os.path.join(TEST_DIR, "real")
    fake_dir = os.path.join(TEST_DIR, "fake")

    real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('jpg','png'))]
    fake_paths = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('jpg','png'))]

    paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)
    return paths, labels

# 图像预处理（和摄像头完全一致）
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)

    # MSR 增强分支
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_img = np.expand_dims(new_img, axis=-1)
    new_img = automatedMSRCR(new_img, [10, 20, 30])
    new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)

    img = img / 255.0
    new_img = new_img / 255.0
    return img, new_img

# 批量预测
def predict():
    paths, labels = get_test_paths()
    y_true = []
    y_pred = []

    print("🔍 正在测试 CASIA-FASD 测试集...")
    for path, label in tqdm(zip(paths, labels), total=len(paths)):
        frame = cv2.imread(path)
        img, msr_img = preprocess(frame)

        pred = model.predict([np.expand_dims(img, 0), np.expand_dims(msr_img, 0)], verbose=0)
        score = pred[0][0]

        y_true.append(label)
        y_pred.append(score)

    return np.array(y_true), np.array(y_pred)

# 计算 EER
def compute_eer(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100

# 主函数
if __name__ == "__main__":
    y_true, y_pred = predict()
    eer = compute_eer(y_true, y_pred)

    # 其他常用指标
    auc = roc_auc_score(y_true, y_pred)
    y_pred_class = (np.array(y_pred) > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_class)
    prec = precision_score(y_true, y_pred_class, zero_division=0)
    rec = recall_score(y_true, y_pred_class, zero_division=0)
    f1 = f1_score(y_true, y_pred_class, zero_division=0)

    # 输出结果
    print("\n" + "="*60)
    print("📊 CASIA-FASD 测试集最终结果")
    print("="*60)
    print(f"✅ EER           = {eer:.4f} %")
    print(f"✅ AUC           = {auc:.4f}")
    print(f"✅ Accuracy      = {acc:.4f}")
    print(f"✅ Precision     = {prec:.4f}")
    print(f"✅ Recall        = {rec:.4f}")
    print(f"✅ F1-Score      = {f1:.4f}")
    print(f"✅ 测试样本总数   = {len(y_true)}")
    print(f"✅ 真实样本       = {np.sum(y_true == 0)}")
    print(f"✅ 攻击样本       = {np.sum(y_true == 1)}")
    print("="*60)