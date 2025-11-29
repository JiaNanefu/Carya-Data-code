import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET

IMAGE_DIR = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\Scaled\images"
LABEL_DIR = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\Scaled\labels\YOLO"
VOC_OUTPUT_DIR = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\Scaled\labels\VOC"
IMG_EXTS = [".jpg", ".jpeg", ".png"]
CLASS_ID_TO_NAME = {
    0: "maturity1",
    1: "maturity2",
    2: "maturity3",
    3: "blurring"
}

def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def yolo_to_voc_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            try:
                cls_id = int(float(parts[0]))
                x_c, y_c, w, h = map(float, parts[1:])

                if cls_id not in CLASS_ID_TO_NAME:
                    continue

                x_c *= img_w
                y_c *= img_h
                w *= img_w
                h *= img_h

                xmin = x_c - w / 2
                ymin = y_c - h / 2
                xmax = x_c + w / 2
                ymax = y_c + h / 2

                xmin = max(0, min(img_w - 1, xmin))
                ymin = max(0, min(img_h - 1, ymin))
                xmax = max(0, min(img_w - 1, xmax))
                ymax = max(0, min(img_h - 1, ymax))

                xmin_int = int(round(xmin))
                ymin_int = int(round(ymin))
                xmax_int = int(round(xmax))
                ymax_int = int(round(ymax))

                boxes.append((CLASS_ID_TO_NAME[cls_id], xmin_int, ymin_int, xmax_int, ymax_int))
            
            except ValueError:
                continue

    return boxes

def save_voc_xml(image_path, boxes, out_xml_path):
    img = imread_unicode(image_path)
    if img is None:
        print(f"[VOC] 读取图像失败：{image_path}")
        return

    h, w, c = img.shape
    filename = os.path.basename(image_path)

    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.basename(os.path.dirname(image_path))

    filename_el = ET.SubElement(annotation, "filename")
    filename_el.text = filename

    size = ET.SubElement(annotation, "size")
    width_el = ET.SubElement(size, "width")
    width_el.text = str(w)
    height_el = ET.SubElement(size, "height")
    height_el.text = str(h)
    depth_el = ET.SubElement(size, "depth")
    depth_el.text = str(c)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for cls_name, xmin, ymin, xmax, ymax in boxes:
        obj = ET.SubElement(annotation, "object")

        name_el = ET.SubElement(obj, "name")
        name_el.text = cls_name

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        xmin_el = ET.SubElement(bndbox, "xmin")
        xmin_el.text = str(xmin)
        ymin_el = ET.SubElement(bndbox, "ymin")
        ymin_el.text = str(ymin)
        xmax_el = ET.SubElement(bndbox, "xmax")
        xmax_el.text = str(xmax)
        ymax_el = ET.SubElement(bndbox, "ymax")
        ymax_el.text = str(ymax)

    tree = ET.ElementTree(annotation)
    os.makedirs(os.path.dirname(out_xml_path), exist_ok=True)
    tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)

def convert_yolo_to_voc():
    print(f"=== 开始转换 YOLO -> VOC ===")
    os.makedirs(VOC_OUTPUT_DIR, exist_ok=True)

    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, f"*{ext}")))

    print(f"[VOC] 共发现 {len(img_paths)} 张图片，准备处理...")

    count = 0
    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, base + ".txt")

        img = imread_unicode(img_path)
        if img is None:
            print(f"[VOC] 跳过（无法读取图像）：{img_path}")
            continue
        h, w, _ = img.shape

        boxes = yolo_to_voc_boxes(label_path, w, h)

        out_xml_path = os.path.join(VOC_OUTPUT_DIR, base + ".xml")
        save_voc_xml(img_path, boxes, out_xml_path)
        count += 1

    print(f"=== 转换完成 ===")
    print(f"已生成 {count} 个 XML 文件到: {VOC_OUTPUT_DIR}")


if __name__ == "__main__":
    convert_yolo_to_voc()