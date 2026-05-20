import os
import cv2
import shutil

# ================== 配置路径（改成你自己的！）==================
# 你的CASIA数据集根目录
CASIA_ROOT = r"F:\数据集\CASIA-FASD\CASIA-FASD"
# 输出目录（你的代码会从这里读数据）
OUTPUT_ROOT = r"E:\Python\Attention-Based_Two-Stream_Convolutional_Networks_for_Face_Spoofing_Detection\CASIA"

# 视频转图片的配置
FRAME_INTERVAL = 10  # 每10帧保存一张图（避免重复）
TARGET_SIZE = (224, 224)  # 模型输入尺寸


# ==============================================================

def prepare_folders(output_dir):
    """创建代码需要的目录结构"""
    splits = ['train', 'test']
    types = ['real', 'fake']
    for split in splits:
        for t in types:
            os.makedirs(os.path.join(output_dir, split, t), exist_ok=True)
    print("✅ 已创建目录结构：", output_dir)


def is_real_video(video_name):
    """判断视频是否为真实人脸（CASIA规则：1-6为real，7+为fake）"""
    base_name = os.path.basename(video_name).lower()
    if base_name.startswith("hr_"):
        return False
    try:
        num = int(base_name.split('.')[0])
        return 1 <= num <= 6
    except:
        return False


def extract_frames(video_path, output_dir, frame_interval=10):
    """从视频中按间隔提取帧并保存"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 无法打开视频：{video_path}")
        return 0

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.resize(frame, TARGET_SIZE)
            save_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_{count}.jpg")
            cv2.imwrite(save_path, frame)
            saved += 1
        count += 1
    cap.release()
    return saved


def process_split(split_name, split_dir, output_root):
    """处理train或test文件夹下的所有视频"""
    if not os.path.exists(split_dir):
        print(f"⚠️ 不存在目录：{split_dir}")
        return

    print(f"\n===== 正在处理 {split_name} 集 =====")
    real_count = 0
    fake_count = 0

    for person_folder in os.listdir(split_dir):
        person_path = os.path.join(split_dir, person_folder)
        if not os.path.isdir(person_path):
            continue

        for video_file in os.listdir(person_path):
            if not video_file.endswith(".avi"):
                continue
            video_path = os.path.join(person_path, video_file)
            is_real = is_real_video(video_file)

            if is_real:
                target_dir = os.path.join(output_root, split_name, "real")
                real_count += extract_frames(video_path, target_dir, FRAME_INTERVAL)
            else:
                target_dir = os.path.join(output_root, split_name, "fake")
                fake_count += extract_frames(video_path, target_dir, FRAME_INTERVAL)

    print(f"✅ {split_name} 集处理完成：real={real_count}张, fake={fake_count}张")


if __name__ == "__main__":
    # 1. 创建目录
    prepare_folders(OUTPUT_ROOT)

    # 2. 处理训练集
    train_dir = os.path.join(CASIA_ROOT, "train_release")
    process_split("train", train_dir, OUTPUT_ROOT)

    # 3. 处理测试集
    test_dir = os.path.join(CASIA_ROOT, "test_release")
    process_split("test", test_dir, OUTPUT_ROOT)

    print("\n🎉 CASIA数据集转换完成！结构如下：")
    print(f"{OUTPUT_ROOT}/")
    print("├── train/")
    print("│   ├── real/  # 真实人脸图片")
    print("│   └── fake/  # 攻击人脸图片")
    print("└── test/")
    print("    ├── real/")
    print("    └── fake/")