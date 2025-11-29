import json
import os
from pathlib import Path
from PIL import Image
from datetime import datetime

def yolo_to_coco(dataset_root, splits=['train', 'val', 'test'], class_names=None):
    dataset_root = Path(dataset_root)
    output_dir = dataset_root / 'coco_annotations'
    output_dir.mkdir(exist_ok=True)
    
    if class_names is None:
        classes_file = dataset_root / 'train' / 'labels' / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(class_names)} classes from {classes_file}")
        else:
            class_names = ['Carya']
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        images_dir = dataset_root / split / 'images'
        labels_dir = dataset_root / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Warning: {split} split not found, skipping...")
            continue
        
        coco_dict = {
            "info": {
                "description": "Carya Dataset - YOLO to COCO Conversion",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y/%m/%d")
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
        
        annotation_id = 1
        image_id = 1
        
        image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
        
        print(f"Found {len(image_files)} images")

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading image {img_path.name}: {e}")
                continue
            
            coco_dict["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
            
            label_path = labels_dir / (img_path.stem + '.txt')
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid annotation in {label_path.name}: {line}")
                        continue
                    
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    bbox_width = float(parts[3])
                    bbox_height = float(parts[4])
                    
                    x_min = (center_x - bbox_width / 2) * width
                    y_min = (center_y - bbox_height / 2) * height
                    abs_width = bbox_width * width
                    abs_height = bbox_height * height
                    
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

            image_id += 1
        
        output_file = output_dir / f'instances_{split}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] {split} split converted successfully!")
        print(f"  - Images: {len(coco_dict['images'])}")
        print(f"  - Annotations: {len(coco_dict['annotations'])}")
        print(f"  - Categories: {len(coco_dict['categories'])}")
        print(f"  - Output file: {output_file}")

if __name__ == '__main__':
    dataset_root = Path(__file__).parent
    
    print("="*60)
    print("YOLO to COCO Format Conversion")
    print("="*60)
    print(f"Dataset root: {dataset_root}")
    
    yolo_to_coco(dataset_root, splits=['train', 'val', 'test'])
    
    print("\n" + "="*60)
    print("Conversion completed!")
    print("="*60)
