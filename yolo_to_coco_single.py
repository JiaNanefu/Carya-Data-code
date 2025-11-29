import json
import os
from pathlib import Path
from PIL import Image
from datetime import datetime

IMAGES_DIR = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\Scaled\images"
LABELS_DIR = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\Scaled\labels\YOLO"
OUTPUT_JSON = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\Scaled\labels\COCO\annotations.json"
CLASS_NAMES = [
    "maturity1",
    "maturity2",
    "maturity3",
    "blurring",
]
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

def yolo_to_coco_single_folder(images_dir, labels_dir, output_json, class_names):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_json = Path(output_json)
    
    print("="*60)
    print("YOLO 转 COCO 格式转换")
    print("="*60)
    print(f"图片文件夹: {images_dir}")
    print(f"标签文件夹: {labels_dir}")
    print(f"输出文件: {output_json}")
    print(f"类别数量: {len(class_names)}")
    print("="*60)
    
    if not images_dir.exists():
        print(f"错误: 图片文件夹不存在 - {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"错误: 标签文件夹不存在 - {labels_dir}")
        return
    
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print("\n类别信息:")
    
    coco_dict = {
        "info": {
            "description": "YOLO to COCO Format Conversion",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    for idx, class_name in enumerate(class_names):
        coco_dict["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "object"
        })
        print(f"  - ID {idx}: {class_name}")
    
    annotation_id = 1
    image_id = 1
    
    print(f"\n发现 {len(class_names)} 个类别")
    print("\n开始处理...\n")
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    image_files = sorted(set(image_files))
    
    processed_count = 0
    skipped_count = 0
    total_annotations = 0
    
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"[错误] 无法读取图片 {img_path.name}: {e}")
            skipped_count += 1
            processed_count += 1
            continue
        
        coco_dict["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })
        
        label_path = labels_dir / (img_path.stem + '.txt')
        
        annotations_in_image = 0
        
        if label_path.exists():
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"[警告] {label_path.name} 中存在无效标注: {line}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])
                        
                        if class_id < 0 or class_id >= len(class_names):
                            print(f"[警告] {label_path.name} 中存在无效类别ID: {class_id}")
                            continue
                        
                        x_min = (center_x - bbox_width / 2) * width
                        y_min = (center_y - bbox_height / 2) * height
                        abs_width = bbox_width * width
                        abs_height = bbox_height * height
                        
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        abs_width = min(abs_width, width - x_min)
                        abs_height = min(abs_height, height - y_min)
                        
                        area = abs_width * abs_height

                        coco_dict["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, abs_width, abs_height],
                            "area": area,
                            "iscrowd": 0
                        })
                        
                        annotation_id += 1
                        annotations_in_image += 1
                        total_annotations += 1
                    
                    except ValueError as e:
                        print(f"[警告] {label_path.name} 中存在格式错误: {line} - {e}")
                        continue
            
            except Exception as e:
                print(f"[错误] 读取标签文件失败 {label_path.name}: {e}")

        processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"已处理: {processed_count}/{len(image_files)} 张图片, 标注数: {total_annotations}")

        image_id += 1
    
    print("\n保存 COCO 格式文件...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_dict, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)
    print(f"成功处理: {processed_count} 张图片")
    print(f"跳过: {skipped_count} 张图片")
    print(f"总标注数: {total_annotations}")
    print(f"类别数: {len(coco_dict['categories'])}")
    print(f"输出文件: {output_json}")
    print(f"文件大小: {output_json.stat().st_size / 1024:.2f} KB")
    print("="*60)


if __name__ == "__main__":
    yolo_to_coco_single_folder(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_json=OUTPUT_JSON,
        class_names=CLASS_NAMES
    )
