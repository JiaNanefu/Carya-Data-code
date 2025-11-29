import os
import glob
import xml.etree.ElementTree as ET

VOC_LABEL_DIR = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\RAWLABELS"
YOLO_OUTPUT_DIR = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\RAWLABELS_txt"
CLASS_NAME_TO_ID = {
    "maturity1": 0,
    "maturity2": 1,
    "maturity3": 2,
    "blurring": 3,
}

def parse_voc_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find("size")
        if size is None:
            print(f"[警告] {xml_path} 缺少 <size> 标签")
            return None, None, []
        
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        boxes = []
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None:
                continue
            
            cls_name = name_el.text
            if cls_name not in CLASS_NAME_TO_ID:
                print(f"[警告] 未知类别: {cls_name}，跳过")
                continue
            
            cls_id = CLASS_NAME_TO_ID[cls_name]
            
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            boxes.append((cls_id, xmin, ymin, xmax, ymax))
        
        return width, height, boxes
    
    except Exception as e:
        print(f"[错误] 解析 XML 失败: {xml_path}, {e}")
        return None, None, []


def voc_to_yolo_boxes(width, height, boxes):
    yolo_boxes = []

    for cls_id, xmin, ymin, xmax, ymax in boxes:
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        
        x_center /= width
        y_center /= height
        w /= width
        h /= height
        
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        
        yolo_boxes.append((cls_id, x_center, y_center, w, h))
    
    return yolo_boxes


def save_yolo_txt(yolo_boxes, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for cls_id, x_c, y_c, w, h in yolo_boxes:
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def convert_voc_to_yolo():
    print(f"=== 开始转换 VOC -> YOLO ===")
    os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)
    
    xml_paths = glob.glob(os.path.join(VOC_LABEL_DIR, "*.xml"))
    print(f"[YOLO] 共发现 {len(xml_paths)} 个 XML 文件，准备处理...")
    
    count = 0
    for xml_path in xml_paths:
        width, height, boxes = parse_voc_xml(xml_path)
        
        if width is None or height is None:
            print(f"[YOLO] 跳过（无法解析）：{xml_path}")
            continue
        
        yolo_boxes = voc_to_yolo_boxes(width, height, boxes)

        base = os.path.splitext(os.path.basename(xml_path))[0]
        output_path = os.path.join(YOLO_OUTPUT_DIR, base + ".txt")
        save_yolo_txt(yolo_boxes, output_path)
        
        count += 1
        
        if count % 100 == 0:
            print(f"[YOLO] 已处理 {count}/{len(xml_paths)} 个文件...")
    
    print(f"=== 转换完成 ===")
    print(f"已生成 {count} 个 YOLO 标签文件到: {YOLO_OUTPUT_DIR}")


if __name__ == "__main__":
    convert_voc_to_yolo()
