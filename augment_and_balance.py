import os
import cv2
import glob
import random
import shutil
import numpy as np
from collections import Counter

import albumentations as A


# ========== 1. 路径和基本参数配置 ==========

# 原始训练集（你现在 996 张图像所在的目录）
SOURCE_IMAGE_DIR = r"C:\Users\20433\Desktop\Hickory_Data_new\3.划分好的数据集（6：2：2）\train\images"
SOURCE_LABEL_DIR = r"C:\Users\20433\Desktop\Hickory_Data_new\3.划分好的数据集（6：2：2）\train\labels"

# 原始验证集（未增强，保持原样）
VAL_IMAGE_DIR = r"C:\Users\20433\Desktop\Hickory_Data_new\3.划分好的数据集（6：2：2）\val\images"
VAL_LABEL_DIR = r"C:\Users\20433\Desktop\Hickory_Data_new\3.划分好的数据集（6：2：2）\val\labels"

# 原始测试集（未增强，保持原样）
TEST_IMAGE_DIR = r"C:\Users\20433\Desktop\Hickory_Data_new\3.划分好的数据集（6：2：2）\test\images"
TEST_LABEL_DIR = r"C:\Users\20433\Desktop\Hickory_Data_new\3.划分好的数据集（6：2：2）\test\labels"

# 增强后新的训练集输出目录（会生成 images / labels 子目录）
OUTPUT_DIR = r"C:\Users\20433\Desktop\Hickory_Data_new\4.数据增强后的训练集"

# 最终整合后的新数据集根目录（包含 train/val/test 三个子目录）
NEW_DATASET_ROOT = r"C:\Users\20433\Desktop\Hickory_Data_new\5.数据增强后的数据集"

# 图像和标签后缀
IMAGE_EXT = ".jpg"   # 如果是 .png 就改掉
LABEL_EXT = ".txt"   # YOLO 格式

# 目标类别数（0,1,2）
NUM_CLASSES = 3

# 目标图片数量（大约 3000）
TARGET_NUM_IMAGES = 3000
MIN_NUM_IMAGES = 2800   # 搜索时允许的最小图像数
MAX_NUM_IMAGES = 3200   # 搜索时允许的最大图像数

# 随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ========== 2. 增强管道：包含所有要求的操作，每个有概率 ==========

def get_augmentation_pipeline():
    """
    返回两个增强管道：
    1) aug_with_bbox: 适用于有 bbox 的图像（包含 RandomSizedBBoxSafeCrop）
    2) aug_no_bbox  : 适用于没有 bbox 的图像（去掉依赖 bbox 的裁剪）
    为了兼容你的 albumentations 版本，只保留最基础的参数，避免警告。
    """
    aug_with_bbox = A.Compose(
        [
            # 1. 随机水平翻转
            A.HorizontalFlip(p=0.5),

            # 2. 随机裁剪 + 缩放到固定尺寸（bbox 安全版本）
            A.RandomSizedBBoxSafeCrop(
                height=640,
                width=640,
                erosion_rate=0.1,
                p=0.3
            ),

            # 3. 随机仿射变换（旋转 + 平移 + 缩放）
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                p=0.5
            ),

            # 4. 光照与色彩抖动
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.7
            ),

            # 5. 模糊（运动模糊 / 高斯模糊 二选一）
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                ],
                p=0.3
            ),

            # 6. 高斯噪声（使用默认参数，避免不兼容）
            A.GaussNoise(p=0.3),

            # 7. Cutout / 随机擦除（使用默认参数，避免不兼容）
            A.CoarseDropout(p=0.4),
        ],
        bbox_params=A.BboxParams(
            format="yolo",              # YOLO 格式 [x_c, y_c, w, h]，0~1
            label_fields=["class_labels"],
            min_visibility=0.0,         # 尽量不因可见度丢 bbox
            clip=True                   # 自动裁剪到 [0,1]
        )
    )

    # 无 bbox 图像用的版本（不需要 RandomSizedBBoxSafeCrop）
    aug_no_bbox = A.Compose(
        [
            A.HorizontalFlip(p=0.5),

            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                p=0.5
            ),

            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.7
            ),

            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                ],
                p=0.3
            ),

            A.GaussNoise(p=0.3),

            A.CoarseDropout(p=0.4),
        ]
    )

    return aug_with_bbox, aug_no_bbox


# ========== 3. 工具函数：加载 / 保存图像和 YOLO 标签 ==========

def read_image_unicode(path):
    """支持中文路径读取图像，返回 RGB 格式"""
    data = np.fromfile(path, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def write_image_unicode(path, image_rgb):
    """支持中文路径保存图像（JPEG），image_rgb 为 RGB"""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(IMAGE_EXT, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise RuntimeError(f"imencode failed for {path}")
    encoded.tofile(path)


def load_yolo_label(label_path, num_classes=3):
    """
    读取 YOLO 格式标签：
    每行：class x_c y_c w h（全部为 0~1 范围的浮点数）
    返回：
        bboxes: [[x_c, y_c, w, h], ...]
        labels: [class_id, ...]
        class_counts: [count0, count1, count2]
    """
    bboxes = []
    labels = []
    class_counts = [0] * num_classes

    if not os.path.exists(label_path):
        return bboxes, labels, class_counts

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= num_classes:
                    continue
                x_c, y_c, w, h = map(float, parts[1:])

                x_c = float(np.clip(x_c, 0.0, 1.0))
                y_c = float(np.clip(y_c, 0.0, 1.0))
                w   = float(np.clip(w,   0.0, 1.0))
                h   = float(np.clip(h,   0.0, 1.0))

                if w <= 0.0 or h <= 0.0:
                    continue

                bboxes.append([x_c, y_c, w, h])
                labels.append(cls_id)
                class_counts[cls_id] += 1

            except Exception:
                continue

    return bboxes, labels, class_counts


def save_yolo_label(label_path, bboxes, labels):
    """
    保存 YOLO 标签，自动裁剪到 [0,1]。
    强制将 cls_id 转成 int，避免写出 0.0 这种格式。
    """
    with open(label_path, "w", encoding="utf-8") as f:
        for bbox, cls_id in zip(bboxes, labels):
            try:
                cls_int = int(cls_id)
            except Exception:
                cls_int = int(float(cls_id))

            x_c, y_c, w, h = bbox
            x_c = float(np.clip(x_c, 0.0, 1.0))
            y_c = float(np.clip(y_c, 0.0, 1.0))
            w   = float(np.clip(w,   0.0, 1.0))
            h   = float(np.clip(h,   0.0, 1.0))

            if w <= 0.0 or h <= 0.0:
                continue

            f.write(f"{cls_int} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def scan_object_counts(label_dir, num_classes=3):
    """统计给定目录下所有 YOLO 标签文件的类别计数"""
    counts = Counter()
    label_paths = glob.glob(os.path.join(label_dir, "*" + LABEL_EXT))
    for lp in label_paths:
        bboxes, labels, _ = load_yolo_label(lp, num_classes)
        for cls_id in labels:
            counts[cls_id] += 1
    return counts


def print_statistics(counts, title):
    total = sum(counts.values())
    print(f"\n--- {title} ---")
    if total == 0:
        print("  没有统计到任何目标。")
        return
    print(f"  总目标数: {total}")
    for cls_id in sorted(counts.keys()):
        cnt = counts[cls_id]
        pct = cnt / total * 100
        print(f"  类别 {cls_id}: {cnt} 个 ({pct:.2f}%)")


# ========== 4. 构建 image_infos & 池分类 & 搜索复制倍数 ==========

def build_image_infos():
    """
    遍历 SOURCE_LABEL_DIR，收集每张图的信息：
        - image_path
        - label_path
        - basename
        - class_counts: [c0, c1, c2]
    同时构建三个 pool：
        pool 2: 含类 2
        pool 1: 含类 1 但不含 2
        pool 0: 其余
    以及每个 pool 内各类的目标总数。
    """
    image_infos = []
    pools = {0: [], 1: [], 2: []}
    pool_obj_counts = {
        0: [0] * NUM_CLASSES,
        1: [0] * NUM_CLASSES,
        2: [0] * NUM_CLASSES,
    }

    label_paths = glob.glob(os.path.join(SOURCE_LABEL_DIR, "*" + LABEL_EXT))
    if not label_paths:
        raise RuntimeError(f"在 {SOURCE_LABEL_DIR} 中没有找到标签文件 (*.txt)。")

    for lp in label_paths:
        base = os.path.splitext(os.path.basename(lp))[0]
        img_path = os.path.join(SOURCE_IMAGE_DIR, base + IMAGE_EXT)
        if not os.path.exists(img_path):
            continue

        bboxes, labels, class_counts = load_yolo_label(lp, NUM_CLASSES)
        info = {
            "basename": base,
            "image_path": img_path,
            "label_path": lp,
            "class_counts": class_counts,
        }
        idx = len(image_infos)
        image_infos.append(info)

        if class_counts[2] > 0:
            pool_id = 2
        elif class_counts[1] > 0:
            pool_id = 1
        else:
            pool_id = 0

        pools[pool_id].append(idx)
        for cls in range(NUM_CLASSES):
            pool_obj_counts[pool_id][cls] += class_counts[cls]

    return image_infos, pools, pool_obj_counts


def search_copy_plan(pools, pool_obj_counts):
    """
    通过暴力搜索找出 (k0, k1, k2)：
      - 总图像数量在 [MIN_NUM_IMAGES, MAX_NUM_IMAGES] 内
      - 各类目标比例都在 30%–40% 之间
      - 总图像数量尽量接近 TARGET_NUM_IMAGES
    """
    best_plan = None
    best_img_diff = None

    n0 = len(pools[0])
    n1 = len(pools[1])
    n2 = len(pools[2])

    print(f"\n共有图片: pool0 = {n0}, pool1 = {n1}, pool2 = {n2}")

    for k0 in range(1, 5):
        for k1 in range(1, 7):
            for k2 in range(1, 9):
                total_images = n0 * k0 + n1 * k1 + n2 * k2
                if total_images < MIN_NUM_IMAGES or total_images > MAX_NUM_IMAGES:
                    continue

                class_totals = [0] * NUM_CLASSES
                for cls in range(NUM_CLASSES):
                    class_totals[cls] = (
                        pool_obj_counts[0][cls] * k0 +
                        pool_obj_counts[1][cls] * k1 +
                        pool_obj_counts[2][cls] * k2
                    )
                total_objects = sum(class_totals)
                if total_objects == 0:
                    continue

                ratios = [ct / total_objects for ct in class_totals]

                if all(0.30 <= r <= 0.40 for r in ratios):
                    diff = abs(total_images - TARGET_NUM_IMAGES)
                    if (best_img_diff is None) or (diff < best_img_diff):
                        best_img_diff = diff
                        best_plan = {
                            "k0": k0,
                            "k1": k1,
                            "k2": k2,
                            "total_images": total_images,
                            "class_totals": class_totals,
                            "ratios": ratios,
                        }

    if best_plan:
        print("\n找到满足条件的复制计划：")
        print(f"  k0 = {best_plan['k0']}, k1 = {best_plan['k1']}, k2 = {best_plan['k2']}")
        print(f"  预计生成图片总数: {best_plan['total_images']} (目标 {TARGET_NUM_IMAGES})")
        print("  预计目标数量及比例：")
        total_obj = sum(best_plan["class_totals"])
        for cls in range(NUM_CLASSES):
            cnt = best_plan["class_totals"][cls]
            pct = cnt / total_obj * 100 if total_obj > 0 else 0
            print(f"    类别 {cls}: {cnt} 个 (~{pct:.2f}%)")
        return best_plan

    print("\n没有找到严格 30%–40% 的方案，放宽条件找最接近 1/3 的。")
    best_plan = None
    best_score = None

    for k0 in range(1, 5):
        for k1 in range(1, 7):
            for k2 in range(1, 9):
                total_images = n0 * k0 + n1 * k1 + n2 * k2
                if total_images < MIN_NUM_IMAGES or total_images > MAX_NUM_IMAGES:
                    continue

                class_totals = [0] * NUM_CLASSES
                for cls in range(NUM_CLASSES):
                    class_totals[cls] = (
                        pool_obj_counts[0][cls] * k0 +
                        pool_obj_counts[1][cls] * k1 +
                        pool_obj_counts[2][cls] * k2
                    )
                total_objects = sum(class_totals)
                if total_objects == 0:
                    continue

                ratios = [ct / total_objects for ct in class_totals]
                score = sum(abs(r - 1/3) for r in ratios)

                if (best_score is None) or (score < best_score):
                    best_score = score
                    best_plan = {
                        "k0": k0,
                        "k1": k1,
                        "k2": k2,
                        "total_images": total_images,
                        "class_totals": class_totals,
                        "ratios": ratios,
                    }

    if best_plan:
        print("\n找到接近 1/3 的复制计划（但可能略超出 30%–40% 范围）：")
        print(f"  k0 = {best_plan['k0']}, k1 = {best_plan['k1']}, k2 = {best_plan['k2']}")
        print(f"  预计生成图片总数: {best_plan['total_images']} (目标 {TARGET_NUM_IMAGES})")
        print("  预计目标数量及比例：")
        total_obj = sum(best_plan["class_totals"])
        for cls in range(NUM_CLASSES):
            cnt = best_plan["class_totals"][cls]
            pct = cnt / total_obj * 100 if total_obj > 0 else 0
            print(f"    类别 {cls}: {cnt} 个 (~{pct:.2f}%)")
        return best_plan

    raise RuntimeError("在给定范围内没有找到合适的复制计划，请适当调整搜索范围。")


# ========== 6. 数据整合相关函数：复制/重命名 train、val、test ==========

def copy_split(src_img_dir, src_lbl_dir, dst_split_root):
    """
    把 src_img_dir / src_lbl_dir 下的文件复制到 dst_split_root/images 和 labels。
    用于 train（保持已有编号）。
    """
    dst_img_dir = os.path.join(dst_split_root, "images")
    dst_lbl_dir = os.path.join(dst_split_root, "labels")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(src_img_dir, "*" + IMAGE_EXT))
    for img_path in img_paths:
        fname = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(dst_img_dir, fname))

    lbl_paths = glob.glob(os.path.join(src_lbl_dir, "*" + LABEL_EXT))
    for lbl_path in lbl_paths:
        fname = os.path.basename(lbl_path)
        shutil.copy2(lbl_path, os.path.join(dst_lbl_dir, fname))

    print(f"  已复制 images: {len(img_paths)}，labels: {len(lbl_paths)} -> {dst_split_root}")


def copy_and_rename_split(src_img_dir, src_lbl_dir, dst_split_root, start_idx):
    """
    把 src_img_dir / src_lbl_dir 下的 (img, txt) 复制到 dst_split_root/images 和 labels，
    并按 start_idx 开始重新编号为 4 位数字（如 3181.jpg, 3181.txt）。
    返回下一个可用的 index（即最后一个编号+1）。
    """
    dst_img_dir = os.path.join(dst_split_root, "images")
    dst_lbl_dir = os.path.join(dst_split_root, "labels")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(src_img_dir, "*" + IMAGE_EXT))
    img_paths = sorted(img_paths)  # 保证稳定顺序

    cur_idx = start_idx
    num_copied = 0

    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(src_lbl_dir, base + LABEL_EXT)

        if not os.path.exists(lbl_path):
            print(f"[警告] {base} 没有找到标签，跳过。")
            continue

        new_name = f"{cur_idx:04d}"

        dst_img_path = os.path.join(dst_img_dir, new_name + IMAGE_EXT)
        dst_lbl_path = os.path.join(dst_lbl_dir, new_name + LABEL_EXT)

        shutil.copy2(img_path, dst_img_path)
        shutil.copy2(lbl_path, dst_lbl_path)

        cur_idx += 1
        num_copied += 1

    print(f"  已复制并重命名 images/labels 共 {num_copied} 对，编号起始为 {start_idx:04d}")
    return cur_idx  # 下一个编号


def merge_splits_to_new_dataset():
    """
    将：
      - 增强后的训练集 OUTPUT_DIR/images, labels（保持原编号）
      - 原验证集 VAL_IMAGE_DIR, VAL_LABEL_DIR（接在 train 后编号）
      - 原测试集 TEST_IMAGE_DIR, TEST_LABEL_DIR（接在 val 后编号）
    复制到 NEW_DATASET_ROOT 下的 train/val/test 三个子目录。
    """
    print("\n=== 开始整合 train/val/test 到新数据集根目录 ===")
    train_root = os.path.join(NEW_DATASET_ROOT, "train")
    val_root = os.path.join(NEW_DATASET_ROOT, "val")
    test_root = os.path.join(NEW_DATASET_ROOT, "test")

    # ------------ 1) 复制增强后的 train（保持原 0001~NNNN 命名） ------------
    enhanced_train_img_dir = os.path.join(OUTPUT_DIR, "images")
    enhanced_train_lbl_dir = os.path.join(OUTPUT_DIR, "labels")

    print("\n[train] 复制增强后的训练集（保持原有编号）...")
    copy_split(enhanced_train_img_dir, enhanced_train_lbl_dir, train_root)

    # 统计 train 有多少张，作为后面 val/test 编号起点
    train_imgs = glob.glob(os.path.join(enhanced_train_img_dir, "*" + IMAGE_EXT))
    n_train = len(train_imgs)
    start_idx = n_train + 1
    print(f"[train] 增强后训练集图片数: {n_train}，val 将从 {start_idx:04d} 开始编号。")

    # ------------ 2) 复制并重命名 val：接在 train 后面 ------------
    print("\n[val] 复制原始验证集，并重新编号（接在 train 之后）...")
    start_idx = copy_and_rename_split(VAL_IMAGE_DIR, VAL_LABEL_DIR, val_root, start_idx)

    # ------------ 3) 复制并重命名 test：接在 val 后面 ------------
    print("\n[test] 复制原始测试集，并重新编号（接在 val 之后）...")
    start_idx = copy_and_rename_split(TEST_IMAGE_DIR, TEST_LABEL_DIR, test_root, start_idx)

    print(f"\n=== 整合完成，新的数据集根目录：{NEW_DATASET_ROOT} ===")
    print(f"最终编号到: {start_idx - 1:04d}")


# ========== 5. 主函数：执行增强、重命名和统计 ==========

def main():
    print("=== 开始自动平衡数据增强 ===")

    # 0. 创建输出目录
    out_img_dir = os.path.join(OUTPUT_DIR, "images")
    out_lbl_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 1. 扫描原始训练集，构建 image_infos 和 pools
    image_infos, pools, pool_obj_counts = build_image_infos()

    # 打印原始统计
    print(f"\n共发现 {len(image_infos)} 张带标签的图片。")

    initial_counts = Counter()
    for info in image_infos:
        cc = info["class_counts"]
        for cls in range(NUM_CLASSES):
            initial_counts[cls] += cc[cls]
    print_statistics(initial_counts, "原始训练集统计")

    # 2. 搜索复制计划 (k0, k1, k2)
    plan = search_copy_plan(pools, pool_obj_counts)
    k0, k1, k2 = plan["k0"], plan["k1"], plan["k2"]

    # 3. 构建 pool -> 复制次数 的映射
    pool_to_k = {0: k0, 1: k1, 2: k2}

    # 4. 准备增强管道（有 bbox / 无 bbox 各一套）
    aug_with_bbox, aug_no_bbox = get_augmentation_pipeline()

    # 5. 开始生成增强数据，并按 0001.jpg / 0001.txt 命名
    global_idx = 1

    for pool_id in [0, 1, 2]:
        indices = pools[pool_id]
        if not indices:
            continue
        copies = pool_to_k[pool_id]

        print(f"\n处理 pool{pool_id}（{len(indices)} 张），每张复制 {copies} 次...")

        for idx in indices:
            info = image_infos[idx]
            img = read_image_unicode(info["image_path"])
            if img is None:
                print(f"  [警告] 无法读取图像: {info['image_path']}")
                continue

            bboxes, labels, _ = load_yolo_label(info["label_path"], NUM_CLASSES)

            for _ in range(copies):
                if bboxes and labels:
                    transformed = aug_with_bbox(
                        image=img,
                        bboxes=bboxes,
                        class_labels=labels
                    )
                    out_img = transformed["image"]
                    out_bboxes = transformed["bboxes"]
                    out_labels = transformed["class_labels"]
                else:
                    transformed = aug_no_bbox(image=img)
                    out_img = transformed["image"]
                    out_bboxes = []
                    out_labels = []

                new_name = f"{global_idx:04d}"
                out_img_path = os.path.join(out_img_dir, new_name + IMAGE_EXT)
                out_lbl_path = os.path.join(out_lbl_dir, new_name + LABEL_EXT)

                try:
                    write_image_unicode(out_img_path, out_img)
                except Exception as e:
                    print(f"  [错误] 保存图像失败 {out_img_path}: {e}")
                    continue

                save_yolo_label(out_lbl_path, out_bboxes, out_labels)

                global_idx += 1

    total_generated = global_idx - 1
    print(f"\n=== 增强完成，共生成 {total_generated} 张图片 ===")

    # 6. 统计新训练集中的类别分布
    final_counts = scan_object_counts(out_lbl_dir, NUM_CLASSES)
    print_statistics(final_counts, "增强后训练集统计")

    total_objects = sum(final_counts.values())
    if total_objects > 0:
        ratios = {cls: final_counts[cls] / total_objects for cls in final_counts}
        if all(0.30 <= ratios[cls] <= 0.40 for cls in ratios):
            print("\n[V] 所有类别的目标比例都在 30%–40% 之间（满足要求）")
        else:
            print("\n[!] 类别比例没有完全落在 30%–40% 之间，但已经根据稀有类自动进行了过采样。")
    else:
        print("\n[警告] 新训练集中没有统计到任何目标，请检查标签是否为 YOLO 归一化格式。")

    print(f"\n增强后的训练集已保存到: {OUTPUT_DIR}")

    # 7. 把增强后的 train + 原 val/test 合并到一个新目录
    merge_splits_to_new_dataset()


if __name__ == "__main__":
    main()
