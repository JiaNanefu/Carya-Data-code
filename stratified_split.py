import os
import shutil
import random
import glob
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

# 您需要配置的参数
# 路径设置
SOURCE_IMAGE_DIR = r"C:\Users\20433\Desktop\Hickory Data\640\images"  # 原始图片文件夹路径
SOURCE_LABEL_DIR = r"C:\Users\20433\Desktop\Hickory Data\640\labels\YOLO"  # 原始标注文件夹路径
OUTPUT_DIR =r"C:\Users\20433\Desktop\Hickory Data\划分后的数据集"  # 划分后数据集的输出根目录

# 文件扩展名
IMAGE_EXTENSION = ".jpg"  # 您的图片扩展名 (例如: .jpg, .png)
LABEL_EXTENSION = ".txt"  # 您的标注扩展名 (例如: .txt, .xml)

# 划分设置
RAREST_CLASS_ID = 2  # 您的最稀有类别的ID (您提到的是 2)
SPLIT_RATIOS = {'train': 0.6, 'val': 0.2, 'test': 0.2}
RANDOM_SEED = 42  # 设置随机种子以便结果可复现

def check_for_rare_class(label_path, rarest_id):
    """
    检查单个标注文件是否包含最稀有类别。
    """

    if LABEL_EXTENSION == ".txt":
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    if class_id == rarest_id:
                        return True
            return False
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            return False

    # === 示例 2: 适用于 PASCAL VOC (.xml) 格式 ===
    # 格式假定为: <object><name>class_name</name>...</object>
    elif LABEL_EXTENSION == ".xml":
        try:
            tree = ET.parse(label_path)
            root = tree.getroot()
            # 假设您的类别 '2' 在xml中存储为字符串 "2" 或其他名称
            # *** 您可能需要将 'str(rarest_id)' 修改为 'your_class_name' ***
            for obj in root.findall('object'):
                name = obj.find('name').text
                # if name == "your_class_name":
                if name == str(rarest_id):
                    return True
            return False
        except Exception as e:
            print(f"Error parsing {label_path}: {e}")
            return False

    else:
        raise ValueError(f"不支持的标签格式: {LABEL_EXTENSION}")


def create_output_dirs(base_dir):
    """
    创建 train/val/test 及其下的 images/labels 文件夹
    """
    for split in ['train', 'val', 'test']:
        for content in ['images', 'labels']:
            path = os.path.join(base_dir, split, content)
            os.makedirs(path, exist_ok=True)


def copy_files(file_basenames, dest_split_name):
    """
    将文件列表复制到目标文件夹
    """
    dest_dir = os.path.join(OUTPUT_DIR, dest_split_name)
    for basename in file_basenames:
        # 构造源文件路径
        img_src_path = os.path.join(SOURCE_IMAGE_DIR, basename + IMAGE_EXTENSION)
        lbl_src_path = os.path.join(SOURCE_LABEL_DIR, basename + LABEL_EXTENSION)

        # 构造目标文件路径
        img_dest_path = os.path.join(dest_dir, 'images', basename + IMAGE_EXTENSION)
        lbl_dest_path = os.path.join(dest_dir, 'labels', basename + LABEL_EXTENSION)

        # 复制文件
        if os.path.exists(img_src_path):
            shutil.copy2(img_src_path, img_dest_path)
        else:
            print(f"警告: 找不到图片 {img_src_path}")

        if os.path.exists(lbl_src_path):
            shutil.copy2(lbl_src_path, lbl_dest_path)
        else:
            print(f"警告: 找不到标签 {lbl_src_path}")



def main():
    #设置随机种子
    random.seed(RANDOM_SEED)

    #创建输出目录
    print("正在创建输出目录...")
    create_output_dirs(OUTPUT_DIR)

    #查找所有标签文件并分离 "稀有" 和 "常规" 池
    pool_rare = []
    pool_common = []

    all_label_files = glob.glob(os.path.join(SOURCE_LABEL_DIR, f"*{LABEL_EXTENSION}"))
    if not all_label_files:
        print(f"错误: 在 {SOURCE_LABEL_DIR} 中未找到 {LABEL_EXTENSION} 标签文件。")
        return

    print(f"正在扫描 {len(all_label_files)} 个标签文件以查找类别 {RAREST_CLASS_ID}...")

    for label_path in all_label_files:
        # 获取不带扩展名的文件名 (basename)
        basename = os.path.splitext(os.path.basename(label_path))[0]

        if check_for_rare_class(label_path, RAREST_CLASS_ID):
            pool_rare.append(basename)
        else:
            pool_common.append(basename)

    print(f"扫描完成：")
    print(f"  包含稀有类 (类别 {RAREST_CLASS_ID}) 的图片: {len(pool_rare)} 张")
    print(f"  不包含稀有类的图片: {len(pool_common)} 张")

    if len(pool_rare) == 0:
        print(f"错误: 稀有池中没有文件。请检查您的 RAREST_CLASS_ID ({RAREST_CLASS_ID}) 是否正确。")
        return

    # 3. 对两个池进行分层划分
    # 需要进行两次 train_test_split 才能得到 train, val, test
    # 第一次: 拆分出 test
    # 第二次: 将剩余部分拆分为 train 和 val

    test_size = SPLIT_RATIOS['test']
    # 计算 val 在 (train + val) 中的相对比例
    val_size_relative = SPLIT_RATIOS['val'] / (SPLIT_RATIOS['train'] + SPLIT_RATIOS['val'])

    # 划分 稀有池
    temp_rare, test_rare = train_test_split(
        pool_rare, test_size=test_size, random_state=RANDOM_SEED
    )
    train_rare, val_rare = train_test_split(
        temp_rare, test_size=val_size_relative, random_state=RANDOM_SEED
    )

    # 划分 常规池
    temp_common, test_common = train_test_split(
        pool_common, test_size=test_size, random_state=RANDOM_SEED
    )
    train_common, val_common = train_test_split(
        temp_common, test_size=val_size_relative, random_state=RANDOM_SEED
    )

    # 4. 合并并打乱最终列表
    train_files = train_rare + train_common
    val_files = val_rare + val_common
    test_files = test_rare + test_common

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    # 5. 打印最终统计数据
    print("\n--- 划分结果 ---")
    print(f"训练集 (Train): {len(train_files)} 张 (其中 {len(train_rare)} 张包含稀有类)")
    print(f"验证集 (Val):   {len(val_files)} 张 (其中 {len(val_rare)} 张包含稀有类)")
    print(f"测试集 (Test):  {len(test_files)} 张 (其中 {len(test_rare)} 张包含稀有类)")
    print(f"总计: {len(train_files) + len(val_files) + len(test_files)} 张")

    # 6. 复制文件到新目录
    print("\n正在复制文件...")
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print("\n--- 任务完成 ---")
    print(f"数据集已成功划分到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()