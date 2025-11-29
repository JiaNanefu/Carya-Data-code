import os
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm


CLASS_TO_MATURITY = {
    0: "maturity1",
    1: "maturity2",
    2: "maturity3",
    3: "blurring"
}


def calculate_image_brightness(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    return float(np.mean(v_channel))


def calculate_roi_brightness(v_channel: np.ndarray, xmin: int, ymin: int, 
                             xmax: int, ymax: int) -> Optional[float]:
    if xmax <= xmin or ymax <= ymin:
        return None
    
    roi = v_channel[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        return None
    
    return float(np.mean(roi))


def parse_yolo_label(label_path: str, img_width: int, img_height: int, 
                     v_channel: np.ndarray) -> List[Dict]:
    instances = []
    
    if not os.path.exists(label_path):
        return instances
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: 无法读取标签文件 {label_path}: {e}")
        return instances
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 5:
            print(f"Warning: 标签文件 {label_path} 第 {line_idx+1} 行格式错误，跳过")
            continue
        
        try:
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError as e:
            print(f"Warning: 标签文件 {label_path} 第 {line_idx+1} 行数值转换失败: {e}")
            continue

        x_c = cx * img_width
        y_c = cy * img_height
        bw = w * img_width
        bh = h * img_height
        
        xmin = int(x_c - bw / 2)
        xmax = int(x_c + bw / 2)
        ymin = int(y_c - bh / 2)
        ymax = int(y_c + bh / 2)

        xmin = max(0, min(xmin, img_width))
        xmax = max(0, min(xmax, img_width))
        ymin = max(0, min(ymin, img_height))
        ymax = max(0, min(ymax, img_height))

        bbox_area_norm = w * h
        bbox_area_pixel = (xmax - xmin) * (ymax - ymin)

        mean_brightness_roi = calculate_roi_brightness(v_channel, xmin, ymin, xmax, ymax)
        if mean_brightness_roi is None:
            print(f"Warning: 标签文件 {label_path} 第 {line_idx+1} 行 ROI 无效，亮度设为 NaN")

        maturity_label = CLASS_TO_MATURITY.get(class_id, f"unknown_{class_id}")
        
        instance = {
            'instance_id': line_idx,
            'class_id': class_id,
            'maturity_label': maturity_label,
            'cx': cx,
            'cy': cy,
            'w': w,
            'h': h,
            'bbox_xmin': xmin,
            'bbox_ymin': ymin,
            'bbox_xmax': xmax,
            'bbox_ymax': ymax,
            'bbox_area_norm': bbox_area_norm,
            'bbox_area_pixel': bbox_area_pixel,
            'mean_brightness_roi': mean_brightness_roi
        }
        
        instances.append(instance)
    
    return instances


def load_image_metadata(image_path: str, labels_root: str, 
                       image_id: int) -> Tuple[Dict, List[Dict]]:
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Warning: 读取图像失败 {image_path}: {e}")
        img = None
    
    if img is None:
        print(f"Warning: 无法读取图像 {image_path}，跳过")
        return None, None
    
    height, width = img.shape[:2]
    file_name = os.path.basename(image_path)
    
    mean_brightness_image = calculate_image_brightness(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    base_name = os.path.splitext(file_name)[0]
    label_path = os.path.join(labels_root, base_name + '.txt')

    instances = parse_yolo_label(label_path, width, height, v_channel)

    fruit_count = len(instances)
    num_maturity1 = sum(1 for inst in instances if inst['class_id'] == 0)
    num_maturity2 = sum(1 for inst in instances if inst['class_id'] == 1)
    num_maturity3 = sum(1 for inst in instances if inst['class_id'] == 2)
    num_blurring = sum(1 for inst in instances if inst['class_id'] == 3)

    image_metadata = {
        'image_id': image_id,
        'file_name': file_name,
        'image_path': image_path,
        'subset': 'all',
        'width': width,
        'height': height,
        'fruit_count': fruit_count,
        'num_maturity1': num_maturity1,
        'num_maturity2': num_maturity2,
        'num_maturity3': num_maturity3,
        'num_blurring': num_blurring,
        'mean_brightness_image': mean_brightness_image
    }

    for inst in instances:
        inst['image_id'] = image_id
        inst['file_name'] = file_name
    
    return image_metadata, instances


def scan_dataset(images_root: str, labels_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_root, ext)))

    image_paths = list(set(image_paths))
    image_paths.sort()
    
    print(f"找到 {len(image_paths)} 张图像")
    
    image_metadata_list = []
    instance_metadata_list = []

    for image_id, image_path in enumerate(tqdm(image_paths, desc="处理图像")):
        img_meta, instances = load_image_metadata(image_path, labels_root, image_id)
        
        if img_meta is None:
            continue
        
        image_metadata_list.append(img_meta)
        instance_metadata_list.extend(instances)

    image_df = pd.DataFrame(image_metadata_list)
    instance_df = pd.DataFrame(instance_metadata_list)

    if not image_df.empty:
        image_df = image_df[[
            'image_id', 'file_name', 'image_path', 'subset', 
            'width', 'height', 'fruit_count',
            'num_maturity1', 'num_maturity2', 'num_maturity3', 'num_blurring',
            'mean_brightness_image'
        ]]
    
    if not instance_df.empty:
        instance_df = instance_df[[
            'image_id', 'file_name', 'instance_id', 'class_id', 'maturity_label',
            'cx', 'cy', 'w', 'h',
            'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
            'bbox_area_norm', 'bbox_area_pixel', 'mean_brightness_roi'
        ]]
    
    return image_df, instance_df


def print_summary(image_df: pd.DataFrame, instance_df: pd.DataFrame):
    print("\n" + "="*60)
    print("数据集统计摘要")
    print("="*60)
    
    if not image_df.empty:
        print(f"总图像数: {len(image_df)}")
        print(f"总实例数: {len(instance_df)}")
        print(f"\n平均每张图像的实例数: {image_df['fruit_count'].mean():.2f}")
        print(f"最大实例数: {image_df['fruit_count'].max()}")
        print(f"最小实例数: {image_df['fruit_count'].min()}")
        
        print(f"\n成熟度分布:")
        print(f"  maturity1 (class 0): {image_df['num_maturity1'].sum()} 个实例")
        print(f"  maturity2 (class 1): {image_df['num_maturity2'].sum()} 个实例")
        print(f"  maturity3 (class 2): {image_df['num_maturity3'].sum()} 个实例")
        print(f"  blurring (class 3): {image_df['num_blurring'].sum()} 个实例")
        
        print(f"\n图像分辨率统计:")
        print(f"  宽度范围: {image_df['width'].min()} - {image_df['width'].max()}")
        print(f"  高度范围: {image_df['height'].min()} - {image_df['height'].max()}")
        
        print(f"\n平均图像亮度: {image_df['mean_brightness_image'].mean():.2f}")
    
    if not instance_df.empty:
        print(f"\n实例边界框统计:")
        print(f"  归一化面积平均值: {instance_df['bbox_area_norm'].mean():.4f}")
        print(f"  像素面积平均值: {instance_df['bbox_area_pixel'].mean():.2f}")
        
        valid_roi_brightness = instance_df['mean_brightness_roi'].dropna()
        if len(valid_roi_brightness) > 0:
            print(f"  平均 ROI 亮度: {valid_roi_brightness.mean():.2f}")
    
    print("="*60 + "\n")


def main():
    base_dir = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\Scaled"
    images_root = os.path.join(base_dir, "images")
    labels_root = os.path.join(base_dir, "labels", "YOLO")

    output_dir = r"C:\Users\20433\Desktop\各种代码\元数据文件"
    os.makedirs(output_dir, exist_ok=True)
    
    image_csv_path = os.path.join(output_dir, "CaryaData_image_metadata.csv")
    instance_csv_path = os.path.join(output_dir, "CaryaData_instance_metadata.csv")

    if not os.path.exists(images_root):
        print(f"错误: 图像目录不存在: {images_root}")
        return
    
    if not os.path.exists(labels_root):
        print(f"错误: 标签目录不存在: {labels_root}")
        return
    
    print(f"数据集路径: {base_dir}")
    print(f"图像目录: {images_root}")
    print(f"标签目录: {labels_root}")
    print(f"输出目录: {output_dir}\n")

    image_df, instance_df = scan_dataset(images_root, labels_root)

    if not image_df.empty:
        image_df.to_csv(image_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n图像级元数据已保存至: {image_csv_path}")
    else:
        print("\n警告: 没有有效的图像数据")
    
    if not instance_df.empty:
        instance_df.to_csv(instance_csv_path, index=False, encoding='utf-8-sig')
        print(f"实例级元数据已保存至: {instance_csv_path}")
    else:
        print("\n警告: 没有有效的实例数据")

    print_summary(image_df, instance_df)


if __name__ == "__main__":
    main()
