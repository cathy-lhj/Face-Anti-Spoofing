import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 导入自定义模块
from attention import attention_model
from datagen import DataGenerator

# 1. 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--bs", default=8, help="batch size")
parser.add_argument("--dim", default=299, help="image dimension")
parser.add_argument("--backbone", default="MobileNetV3", help="backbone architecture")
parser.add_argument("--weight", required=True, help="path to the best weight file (.hdf5)")
args = parser.parse_args()

bs = int(args.bs)
dim = (int(args.dim), int(args.dim))

# 2. 准备测试数据
print("--- Loading Test Data ---")
test_images = []
# 根据你的路径结构，这里指向测试集
test_path = 'CASIA\\test'
for folder in ['fake', 'real']:
    folder_path = os.path.join(test_path, folder)
    for image in os.listdir(folder_path):
        test_images.append(os.path.join(folder_path, image))

test_labels_dict = {}
y_true = []
for image in test_images:
    # 逻辑与 train.py 保持一致: real=0, fake=1
    label = 0 if 'real' in image else 1
    test_labels_dict[image] = label
    y_true.append(label)

# 创建测试生成器 (shuffle需设为False以保证预测顺序一致)
test_gen = DataGenerator(test_images, test_labels_dict, batch_size=bs, dim=dim, type_gen='test')

# 3. 加载模型与权重
print(f"--- Initializing Model: {args.backbone} ---")
model = attention_model(1, backbone=args.backbone, shape=(dim[0], dim[1], 3))
model.load_weights(args.weight)
print(f"Successfully loaded weights from {args.weight}")

# # 4. 执行预测
# print("--- Predicting ---")
# # 使用 predict_generator 获取所有样本的预测概率
# y_pred_prob = model.predict_generator(test_gen, verbose=1)
# # 由于 DataGenerator 可能会为了补齐 batch 而多出样本，截取实际长度
# y_pred_prob = y_pred_prob[:len(y_true)]
# # 概率阈值判定 (通常 0.5)
# y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# 4. 执行预测
print("--- Predicting ---")
y_pred_prob = model.predict_generator(test_gen, verbose=1)

# --- 关键修改：同步 y_true 的长度 ---
# 获取生成器实际参与预测的总样本数 (batch 数 * batch_size)
total_predict_samples = len(y_pred_prob)
y_true = y_true[:total_predict_samples]
# -----------------------------------

# 概率阈值判定
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# 5. 计算评价指标
print("\n" + "="*30)
print("      EVALUATION RESULTS")
print("="*30)

acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1  = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_prob)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {pre:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")

# 6. 计算 EER (Equal Error Rate)
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=1)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(f"EER       : {eer:.4f}")
print("="*30)

# 7. 可视化 ROC 曲线 (可选)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
print(y_pred_prob[:10]) # 打印前10个预测概率