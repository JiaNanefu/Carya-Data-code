#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO标注框质量筛选工具

功能：
- 只分析YOLO标注框内的区域
- 根据质量阈值自动筛选不合格图片
- 输出不合格图片列表和简要报告

依赖：
- opencv-python
- numpy
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse YOLO label file, return list of (class_id, x_center, y_center, w, h)"""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append((cls, xc, yc, w, h))
            except ValueError:
                continue
    return boxes


def yolo_to_xyxy(box: Tuple[int, float, float, float, float], img_w: int, img_h: int) -> Tuple[int, int, int, int, int]:
    """Convert YOLO box to pixel coordinates (class_id, x1, y1, x2, y2)"""
    cls, xc, yc, w, h = box
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    
    # Clamp to image bounds
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    
    return cls, x1, y1, x2, y2


def iter_images(img_dir: Path, exts):
    files = []
    for ext in exts:
        files += list(img_dir.rglob(f"*.{ext}"))
        files += list(img_dir.rglob(f"*.{ext.upper()}"))
    return sorted(set(files))


def imread_bgr(path: Path):
    # robust for non-ascii paths
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def maybe_resize(bgr: np.ndarray, long_side: int):
    if long_side <= 0:
        return bgr
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return bgr
    scale = long_side / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def to_gray_u8(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)


# ---- metrics ----
def laplacian_variance(gray_u8: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    return float(lap.var())


def tenengrad_energy(gray_u8: np.ndarray) -> float:
    gx = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    return float(np.mean(mag2))


def exposure_stats(gray_u8: np.ndarray):
    """计算曝光统计"""
    over = float(np.mean(gray_u8 >= 245))
    under = float(np.mean(gray_u8 <= 10))
    lum_std = float(np.std(gray_u8))
    return over, under, lum_std


def main():
    ap = argparse.ArgumentParser(description="YOLO标注框质量筛选工具")
    ap.add_argument("--img_dir", type=str, required=True, help="图像目录")
    ap.add_argument("--label_dir", type=str, required=True, help="YOLO标注目录")
    ap.add_argument("--out_dir", type=str, default="bad_quality", help="输出目录")
    ap.add_argument("--exts", type=str, default="jpg,jpeg,png", help="图像扩展名")
    
    # 质量阈值
    ap.add_argument("--blur_lap_min", type=float, default=100.0, help="最小Laplacian方差阈值（低于=模糊）")
    ap.add_argument("--blur_ten_min", type=float, default=500.0, help="最小Tenengrad阈值（低于=模糊）")
    ap.add_argument("--overexp_max", type=float, default=0.10, help="最大过曝比例（高于=过曝）")
    ap.add_argument("--underexp_max", type=float, default=0.10, help="最大欠曝比例（高于=欠曝）")
    ap.add_argument("--lum_std_min", type=float, default=25.0, help="最小亮度标准差（低于=低对比度）")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    paths = iter_images(img_dir, exts)
    if not paths:
        raise SystemExit(f"未找到图像文件在 {img_dir}")

    total = len(paths)
    bad_quality_images = {}  # {image_path: [bad_reasons]}
    total_boxes = 0
    bad_boxes = 0
    total_boxes_by_class = {}  # 每个类别的总标注框数
    bad_boxes_by_class = {}    # 每个类别的不合格标注框数
    bad_reasons_by_class = {}  # 每个类别的不合格原因统计 {cls: {reason: count}}
    
    for idx, img_path in enumerate(paths, 1):
        if idx % 50 == 0 or idx == 1:
            print(f"处理中 {idx}/{total} ({100*idx//total}%)...")
        
        try:
            # 查找对应的标注文件
            rel_path = img_path.relative_to(img_dir)
            label_path = label_dir / rel_path.parent / (img_path.stem + '.txt')
            
            if not label_path.exists():
                continue  # 无标注文件，跳过
            
            # 读取图像
            bgr = imread_bgr(img_path)
            img_h, img_w = bgr.shape[:2]
            
            # 解析YOLO标注框
            boxes = parse_yolo_label(label_path)
            if not boxes:
                continue
            
            # 处理每个标注框
            img_bad_reasons = []
            for box_idx, box in enumerate(boxes):
                try:
                    cls, x1, y1, x2, y2 = yolo_to_xyxy(box, img_w, img_h)
                    
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    # 跳过无效框
                    if box_w <= 0 or box_h <= 0:
                        continue
                    
                    # 裁剪ROI
                    roi_bgr = bgr[y1:y2, x1:x2]
                    if roi_bgr.size == 0:
                        continue
                    
                    roi_gray = to_gray_u8(roi_bgr)
                    
                    # 计算核心质量指标
                    lap = laplacian_variance(roi_gray)
                    ten = tenengrad_energy(roi_gray)
                    over, under, lum_std = exposure_stats(roi_gray)
                    
                    total_boxes += 1
                    
                    # 统计类别
                    if cls not in total_boxes_by_class:
                        total_boxes_by_class[cls] = 0
                        bad_boxes_by_class[cls] = 0
                        bad_reasons_by_class[cls] = {}
                    total_boxes_by_class[cls] += 1
                    
                    # 质量检查
                    bad_reasons = []
                    bad_reason_types = []  # 用于统计的原因类型
                    
                    if lap < args.blur_lap_min:
                        bad_reasons.append(f"模糊(lap={lap:.0f})")
                        bad_reason_types.append("模糊_Laplacian")
                    if ten < args.blur_ten_min:
                        bad_reasons.append(f"模糊(ten={ten:.0f})")
                        bad_reason_types.append("模糊_Tenengrad")
                    if over > args.overexp_max:
                        bad_reasons.append(f"过曝({over:.1%})")
                        bad_reason_types.append("过曝")
                    if under > args.underexp_max:
                        bad_reasons.append(f"欠曝({under:.1%})")
                        bad_reason_types.append("欠曝")
                    if lum_std < args.lum_std_min:
                        bad_reasons.append(f"低对比度({lum_std:.0f})")
                        bad_reason_types.append("低对比度")
                    
                    if bad_reasons:
                        bad_boxes += 1
                        bad_boxes_by_class[cls] += 1
                        
                        # 统计该类别的各种不合格原因
                        for reason_type in bad_reason_types:
                            if reason_type not in bad_reasons_by_class[cls]:
                                bad_reasons_by_class[cls][reason_type] = 0
                            bad_reasons_by_class[cls][reason_type] += 1
                        
                        img_bad_reasons.extend(bad_reasons)
                        
                except Exception as e:
                    print(f"  [警告] 处理框 {box_idx} 失败 in {img_path.name}: {e}")
            
            # 如果图像有不合格的框，记录下来
            if img_bad_reasons:
                bad_quality_images[str(img_path)] = img_bad_reasons
                    
        except Exception as e:
            print(f"  [警告] 处理图像失败 {img_path.name}: {e}")

    # 计算类别统计
    class_stats = []
    for cls in sorted(total_boxes_by_class.keys()):
        total_cls = total_boxes_by_class[cls]
        bad_cls = bad_boxes_by_class[cls]
        
        # 整理该类别的不合格原因
        reasons_detail = {}
        if cls in bad_reasons_by_class:
            for reason, count in bad_reasons_by_class[cls].items():
                reasons_detail[reason] = {
                    "数量": count,
                    "占该类别不合格框的比例": f"{100*count/bad_cls:.1f}%" if bad_cls > 0 else "0%",
                    "占该类别总框的比例": f"{100*count/total_cls:.1f}%" if total_cls > 0 else "0%"
                }
        
        class_stats.append({
            "类别ID": cls,
            "总标注框数": total_cls,
            "不合格标注框数": bad_cls,
            "不合格比例": f"{100*bad_cls/total_cls:.1f}%" if total_cls > 0 else "0%",
            "不合格原因详情": reasons_detail
        })
    
    # 保存简要报告
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "处理图像总数": total,
        "标注框总数": total_boxes,
        "不合格标注框数": bad_boxes,
        "不合格标注框比例": f"{100*bad_boxes/total_boxes:.1f}%" if total_boxes > 0 else "0%",
        "涉及不合格图片数": len(bad_quality_images),
        "类别统计": class_stats,
        "质量阈值": {
            "模糊_Laplacian最小值": args.blur_lap_min,
            "模糊_Tenengrad最小值": args.blur_ten_min,
            "过曝_最大比例": f"{args.overexp_max:.0%}",
            "欠曝_最大比例": f"{args.underexp_max:.0%}",
            "低对比度_亮度标准差最小值": args.lum_std_min,
        }
    }
    report_file = out_dir / "report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 复制不合格图片到输出文件夹
    if bad_quality_images:
        bad_dir = out_dir / "images"
        bad_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n正在复制 {len(bad_quality_images)} 张不合格图片到 {bad_dir}...")
        
        for img_path_str in bad_quality_images:
            img_path = Path(img_path_str)
            if img_path.exists():
                dest = bad_dir / img_path.name
                shutil.copy2(img_path, dest)
        
        # 保存不合格图片列表
        bad_list_file = out_dir / "bad_quality_list.txt"
        with open(bad_list_file, "w", encoding="utf-8") as f:
            for img_path, reasons in sorted(bad_quality_images.items()):
                f.write(f"{img_path}\t{', '.join(set(reasons))}\n")
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"✓ 处理完成")
    print(f"  - 处理图像: {total} 张")
    print(f"  - 标注框总数: {total_boxes} 个")
    print(f"  - 不合格标注框: {bad_boxes} 个 ({100*bad_boxes/total_boxes:.1f}%)" if total_boxes > 0 else "")
    print(f"  - 不合格图片: {len(bad_quality_images)} 张")
    
    # 打印类别统计
    if total_boxes_by_class:
        print(f"\n类别质量统计:")
        print(f"  {'类别ID':<8} {'总数':<8} {'不合格':<8} {'不合格率':<10}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for cls in sorted(total_boxes_by_class.keys()):
            total_cls = total_boxes_by_class[cls]
            bad_cls = bad_boxes_by_class[cls]
            rate = f"{100*bad_cls/total_cls:.1f}%" if total_cls > 0 else "0%"
            print(f"  {cls:<8} {total_cls:<8} {bad_cls:<8} {rate:<10}")
        
        # 打印不合格原因分析
        print(f"\n各类别不合格原因详细分析:")
        for cls in sorted(total_boxes_by_class.keys()):
            bad_cls = bad_boxes_by_class[cls]
            if bad_cls > 0 and cls in bad_reasons_by_class:
                print(f"\n  类别 {cls} (不合格框数: {bad_cls}):")
                reasons = bad_reasons_by_class[cls]
                # 按数量降序排列
                sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
                for reason, count in sorted_reasons:
                    percentage = 100 * count / bad_cls
                    print(f"    - {reason:<20} {count:>3}个 ({percentage:>5.1f}% 的不合格框)")
    
    if bad_quality_images:
        print(f"\n✓ 不合格图片已保存到: {bad_dir}")
        print(f"✓ 详细列表: {bad_list_file}")
        print(f"✓ 简要报告: {report_file}")
    else:
        print(f"\n✓ 所有图片质量合格！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
