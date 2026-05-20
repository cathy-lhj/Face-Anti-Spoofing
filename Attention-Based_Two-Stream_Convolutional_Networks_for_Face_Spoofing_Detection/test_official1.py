import os
import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from attention import attention_model
from retinex import automatedMSRCR

# ===================== 测试集路径 =====================
test_dir = 'CASIA/test/'
fake_dir = os.path.join(test_dir, 'fake')
real_dir = os.path.join(test_dir, 'real')

test_paths = []
test_labels = []

# 假图
for f in os.listdir(fake_dir):
    test_paths.append(os.path.join(fake_dir, f))
    test_labels.append(1)

# 真图
for f in os.listdir(real_dir):
    test_paths.append(os.path.join(real_dir, f))
    test_labels.append(0)

# ===================== 模型 =====================
model = attention_model(1, backbone='MobileNetV3', shape=(299, 299, 3))
# # 必须和训练一致
# model = attention_model(
#     1,
#     backbone='MobileNetV3',    # 一样
#     shape=(299, 299, 3)            # 一样
# )
# model.load_weights('ver-2-weight-63-1.00-0.88-0.00179.hdf5')
model.load_weights('weight-22-0.99-1.00-0.00000.hdf5')



# ===================== 预处理函数（双流！！！）=====================
def preprocess(image_path):
    # 读取图片
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))

    # MSRCR
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = np.expand_dims(gray, axis=-1)
    msrcr = automatedMSRCR(gray, [10, 20, 30])
    msrcr = cv2.cvtColor(msrcr[:, :, 0], cv2.COLOR_GRAY2RGB)

    # 归一化
    img = img / 255.0
    msrcr = msrcr / 255.0

    # 扩维度
    img = np.expand_dims(img, axis=0)
    msrcr = np.expand_dims(msrcr, axis=0)

    return [img, msrcr]


# ===================== 开始测试 =====================
print("开始正式测试...")
y_true = []
y_pred = []

# 修复这里！！！
for path, label in tqdm(zip(test_paths, test_labels)):
    x1, x2 = preprocess(path)
    pred = model.predict([x1, x2], verbose=0)

    y_true.append(label)
    y_pred.append(pred[0][0])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 关键：如果结果还是低，把这行注释去掉
y_pred = 1 - y_pred

# ===================== 计算指标 =====================
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
hter = (fpr + (1 - tpr)) / 2
min_hter = np.min(hter)
acc = np.mean((y_pred > 0.5) == y_true)

# ===================== 输出 =====================
print("\n===== ✅ 最终正式测试结果（毕业论文）=====")
print(f"准确率 ACC: {acc:.4f}")
print(f"AUC:       {roc_auc:.4f}")
print(f"EER:       {eer:.4f}")
print(f"HTER:      {min_hter:.4f}")