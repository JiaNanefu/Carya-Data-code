import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# ================== 1. 路径配置 ==================
# 原始图片文件夹
IMAGE_DIR = r"C:\Users\20433\Desktop\Hickory_Data_new\(6)CaryaData_v1_Final_Organize\images\test"
# 原始 YOLO 标签文件夹 (.txt)
LABEL_DIR = r"C:\Users\20433\Desktop\Hickory_Data_new\(6)CaryaData_v1_Final_Organize\labels\YOLO\test"

# VOC XML 输出目录（脚本会自动创建）
VOC_OUTPUT_DIR = r"C:\Users\20433\Desktop\Hickory_Data_new\(6)CaryaData_v1_Final_Organize\labels\VOC\test"

# 支持的图片后缀
IMG_EXTS = [".jpg", ".jpeg", ".png"]

# YOLO 类别 ID -> 类别名称映射
CLASS_ID_TO_NAME = {
    0: "maturity1",
    1: "maturity2",
    2: "maturity3",
}

# ================== 2. 工具函数 ==================

def imread_unicode(path):
    """
    支持包含中文路径的图片读取
    返回: numpy array (OpenCV BGR format)
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

# ================== 3. 核心转换逻辑 (YOLO -> VOC) ==================

def yolo_to_voc_boxes(label_path, img_w, img_h):
    """
    读取单个 YOLO txt 文件，转换为 VOC 格式坐标
    返回: [ (cls_name, xmin, ymin, xmax, ymax), ... ]
    """
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

                # --- 坐标反归一化 ---
                # YOLO (x_center, y_center, w, h) 均为 0~1 相对值
                x_c *= img_w
                y_c *= img_h
                w *= img_w
                h *= img_h

                # 转换为左上角和右下角坐标 (xmin, ymin, xmax, ymax)
                xmin = x_c - w / 2
                ymin = y_c - h / 2
                xmax = x_c + w / 2
                ymax = y_c + h / 2

                # --- 边界截断 (Clip) ---
                # 确保坐标不会超出图片边缘
                xmin = max(0, min(img_w - 1, xmin))
                ymin = max(0, min(img_h - 1, ymin))
                xmax = max(0, min(img_w - 1, xmax))
                ymax = max(0, min(img_h - 1, ymax))

                # 转为整数
                xmin_int = int(round(xmin))
                ymin_int = int(round(ymin))
                xmax_int = int(round(xmax))
                ymax_int = int(round(ymax))

                boxes.append((CLASS_ID_TO_NAME[cls_id], xmin_int, ymin_int, xmax_int, ymax_int))
            
            except ValueError:
                continue

    return boxes


def save_voc_xml(image_path, boxes, out_xml_path):
    """
    根据坐标信息生成并保存 Pascal VOC 格式的 .xml 文件
    """
    img = imread_unicode(image_path)
    if img is None:
        print(f"[VOC] 读取图像失败：{image_path}")
        return

    h, w, c = img.shape
    filename = os.path.basename(image_path)

    # 构建 XML 树结构
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.basename(os.path.dirname(image_path))

    filename_el = ET.SubElement(annotation, "filename")
    filename_el.text = filename

    # 尺寸信息
    size = ET.SubElement(annotation, "size")
    width_el = ET.SubElement(size, "width")
    width_el.text = str(w)
    height_el = ET.SubElement(size, "height")
    height_el.text = str(h)
    depth_el = ET.SubElement(size, "depth")
    depth_el.text = str(c)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    # 遍历每个目标框
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

    # 保存 XML
    tree = ET.ElementTree(annotation)
    os.makedirs(os.path.dirname(out_xml_path), exist_ok=True)
    # xml_declaration=True 确保文件头包含 <?xml version='1.0' encoding='utf-8'?>
    tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)


def convert_yolo_to_voc():
    print(f"=== 开始转换 YOLO -> VOC ===")
    os.makedirs(VOC_OUTPUT_DIR, exist_ok=True)

    # 扫描所有支持后缀的图片
    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, f"*{ext}")))

    print(f"[VOC] 共发现 {len(img_paths)} 张图片，准备处理...")

    count = 0
    for img_path in img_paths:
        # 获取对应的 txt 路径
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, base + ".txt")

        # 读取图像获取宽和高 (W, H)
        img = imread_unicode(img_path)
        if img is None:
            print(f"[VOC] 跳过（无法读取图像）：{img_path}")
            continue
        h, w, _ = img.shape

        # 转换坐标
        boxes = yolo_to_voc_boxes(label_path, w, h)
        
        # 如果没有框，仍然生成一个只有 header 的空 xml（有些数据集需要空 xml 表示负样本）
        if not boxes:
            # print(f"[VOC] {base} 无标注，生成空 XML")
            pass
        
        # 保存
        out_xml_path = os.path.join(VOC_OUTPUT_DIR, base + ".xml")
        save_voc_xml(img_path, boxes, out_xml_path)
        count += 1

    print(f"=== 转换完成 ===")
    print(f"已生成 {count} 个 XML 文件到: {VOC_OUTPUT_DIR}")


if __name__ == "__main__":
    convert_yolo_to_voc()