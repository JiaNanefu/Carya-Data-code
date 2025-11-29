import os
from collections import Counter
import xml.etree.ElementTree as ET

def count_yolo_labels(label_dir):
    counts = Counter()
    total_boxes = 0

    for root, _, files in os.walk(label_dir):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = parts[0]
                    counts[class_id] += 1
                    total_boxes += 1

    return counts, total_boxes


def count_voc_labels(xml_dir):
    counts = Counter()
    total_boxes = 0

    for root, _, files in os.walk(xml_dir):
        for fname in files:
            if not fname.endswith(".xml"):
                continue
            fpath = os.path.join(root, fname)

            try:
                tree = ET.parse(fpath)
                xml_root = tree.getroot()
            except Exception as e:
                print(f"解析出错: {fpath}, 错误: {e}")
                continue

            for obj in xml_root.findall("object"):
                name_node = obj.find("name")
                if name_node is None:
                    continue
                cls_name = name_node.text.strip()
                counts[cls_name] += 1
                total_boxes += 1

    return counts, total_boxes


if __name__ == "__main__":
    mode = "voc"

    if mode == "yolo":
        label_dir = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\labels\YOLO"
        counts, total = count_yolo_labels(label_dir)
        print(f"=== YOLO 标注统计结果（文件夹：{label_dir}）===")
    else:
        xml_dir = r"C:\Users\20433\Desktop\数据集论文材料\CaryaData\labels\VOC"
        counts, total = count_voc_labels(xml_dir)
        print(f"=== VOC 标注统计结果（文件夹：{xml_dir}）===")


    for cls, num in sorted(counts.items(), key=lambda x: str(x[0])):
        print(f"类别 {cls}: {num} 个目标")

    print(f"\n总目标框数: {total}")
    print(f"总类别数: {len(counts)}")
