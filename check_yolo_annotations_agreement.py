#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check strict agreement between two YOLO annotations for the same images.

Logic:
- For each image, compare two annotations (A and B)
- Strict criteria for PASS:
  1. Box count must be the same
  2. All matched boxes must have the same class_id
  3. All matched boxes must have IoU >= 90%
- FAIL handling: Copy image + both labels to 'rejected' folder
- PASS handling: Randomly pick one annotation + image to 'qualified' folder

YOLO Format:
- Each image has a corresponding .txt file with the same name
- Each line: class_id x_center y_center width height (normalized 0-1)

Outputs:
- Qualified dataset with images and labels
- Rejected folder with images and both labels (A and B)
- Statistics report (JSON)
"""

import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


@dataclass
class Box:
    """YOLO format box (normalized coordinates)."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    
    def to_xyxy_normalized(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) normalized format."""
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return (x1, y1, x2, y2)
    
    def to_xyxy_pixel(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) pixel format."""
        x1, y1, x2, y2 = self.to_xyxy_normalized()
        return (
            int(x1 * img_w),
            int(y1 * img_h),
            int(x2 * img_w),
            int(y2 * img_h),
        )


def iou_xyxy(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    """Calculate IoU between two boxes in (x1, y1, x2, y2) format."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def parse_yolo_label(label_path: Path) -> List[Box]:
    """Parse a YOLO format label file."""
    boxes = []
    if not label_path.exists():
        return boxes
    
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append(Box(class_id, x_center, y_center, width, height))
            except ValueError:
                continue
    return boxes


def find_label_files(labels_dir: Path) -> Dict[str, Path]:
    """Find all label files in a directory (recursively), return {stem: path}."""
    label_map = {}
    # Use rglob for recursive search to handle train/val/test subdirectories
    for p in labels_dir.rglob("*.txt"):
        # Skip cache files
        if p.suffix == ".cache" or ".cache" in p.name:
            continue
        label_map[p.stem] = p
    return label_map


def find_image_files(images_dir: Path) -> Dict[str, Path]:
    """Find all image files in a directory (recursively), return {stem: path}."""
    image_map = {}
    for ext in ("jpg", "jpeg", "png", "bmp", "webp", "JPG", "JPEG", "PNG", "BMP", "WEBP"):
        for p in images_dir.rglob(f"*.{ext}"):
            if p.stem not in image_map:
                image_map[p.stem] = p
    return image_map


def match_boxes(
    boxes_a: List[Box],
    boxes_b: List[Box],
    require_same_class: bool = True,
    min_iou_for_match: float = 0.0,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Match boxes between two annotators using Hungarian algorithm or greedy.
    
    Returns:
      matches: list of (idx_a, idx_b, iou)
      unmatched_a: list of idx in A
      unmatched_b: list of idx in B
    """
    nA, nB = len(boxes_a), len(boxes_b)
    if nA == 0 or nB == 0:
        return [], list(range(nA)), list(range(nB))

    # Build IoU matrix using normalized coordinates
    iou_mat = [[0.0 for _ in range(nB)] for _ in range(nA)]
    for i, ba in enumerate(boxes_a):
        for j, bb in enumerate(boxes_b):
            if require_same_class and ba.class_id != bb.class_id:
                iou_mat[i][j] = 0.0
            else:
                v = iou_xyxy(ba.to_xyxy_normalized(), bb.to_xyxy_normalized())
                iou_mat[i][j] = v if v >= min_iou_for_match else 0.0

    matches: List[Tuple[int, int, float]] = []
    used_a, used_b = set(), set()

    if HAS_SCIPY:
        import numpy as np  # type: ignore
        cost = 1.0 - np.array(iou_mat, dtype=float)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind.tolist(), col_ind.tolist()):
            iou = iou_mat[i][j]
            if iou > 0.0:
                matches.append((i, j, iou))
                used_a.add(i)
                used_b.add(j)
    else:
        # Greedy fallback: repeatedly pick best IoU remaining
        candidates = []
        for i in range(nA):
            for j in range(nB):
                if iou_mat[i][j] > 0.0:
                    candidates.append((iou_mat[i][j], i, j))
        candidates.sort(reverse=True)
        for iou, i, j in candidates:
            if i in used_a or j in used_b:
                continue
            matches.append((i, j, iou))
            used_a.add(i)
            used_b.add(j)

    unmatched_a = [i for i in range(nA) if i not in used_a]
    unmatched_b = [j for j in range(nB) if j not in used_b]
    return matches, unmatched_a, unmatched_b


def compute_score(matches: List[Tuple[int, int, float]], nA: int, nB: int) -> Tuple[float, float]:
    """
    Compute agreement score.
    
    score = sum_iou / max(nA, nB)
    mean_iou_matched = sum_iou / max(1, matched_count)
    """
    denom = max(nA, nB)
    if denom == 0:
        return 1.0, 1.0  # both empty => perfect agreement
    sum_iou = sum(m[2] for m in matches)
    score = sum_iou / float(denom)
    mean_iou = sum_iou / float(max(1, len(matches)))
    return score, mean_iou


def draw_viz(
    img_path: Path,
    boxes_a: List[Box],
    boxes_b: List[Box],
    matches: List[Tuple[int, int, float]],
    out_path: Path,
) -> None:
    """Draw visualization overlay for disagreements."""
    if not HAS_CV2:
        return

    img = cv2.imread(str(img_path))
    if img is None:
        return
    
    h, w = img.shape[:2]

    # Colors: A=red, B=green, matched=yellow
    for ba in boxes_a:
        x1, y1, x2, y2 = ba.to_xyxy_pixel(w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"A:{ba.class_id}", (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for bb in boxes_b:
        x1, y1, x2, y2 = bb.to_xyxy_pixel(w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"B:{bb.class_id}", (x1, min(h - 1, y2 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw matched pairs with thicker border & IoU text
    for ia, ib, iou in matches:
        ba, bb = boxes_a[ia], boxes_b[ib]
        ax1, ay1, ax2, ay2 = ba.to_xyxy_pixel(w, h)
        bx1, by1, bx2, by2 = bb.to_xyxy_pixel(w, h)
        ux1, uy1, ux2, uy2 = min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)
        cv2.rectangle(img, (ux1, uy1), (ux2, uy2), (0, 255, 255), 2)
        cv2.putText(img, f"IoU={iou:.3f}", (ux1, max(0, uy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(
        description="Strict agreement check: same count, same class, IoU>=90%"
    )
    ap.add_argument("--labels_a", required=True, type=str, 
                    help="Annotator A labels directory")
    ap.add_argument("--labels_b", required=True, type=str, 
                    help="Annotator B labels directory")
    ap.add_argument("--images_dir", required=True, type=str, 
                    help="Images directory")
    ap.add_argument("--out_dir", required=True, type=str, 
                    help="Output directory for qualified dataset")
    ap.add_argument("--iou_threshold", type=float, default=0.9, 
                    help="IoU threshold for acceptance (default: 0.9)")
    args = ap.parse_args()

    labels_a_dir = Path(args.labels_a)
    labels_b_dir = Path(args.labels_b)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    
    # Create output directories
    qualified_img_dir = out_dir / "images"
    qualified_lbl_dir = out_dir / "labels"
    rejected_img_dir = out_dir / "rejected" / "images"
    rejected_lbl_a_dir = out_dir / "rejected" / "labels_a"
    rejected_lbl_b_dir = out_dir / "rejected" / "labels_b"
    
    for d in [qualified_img_dir, qualified_lbl_dir, rejected_img_dir, 
              rejected_lbl_a_dir, rejected_lbl_b_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Validate directories
    if not labels_a_dir.is_dir():
        raise ValueError(f"Labels A directory not found: {labels_a_dir}")
    if not labels_b_dir.is_dir():
        raise ValueError(f"Labels B directory not found: {labels_b_dir}")
    if not images_dir.is_dir():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Find all label files
    labels_a_map = find_label_files(labels_a_dir)
    labels_b_map = find_label_files(labels_b_dir)
    images_map = find_image_files(images_dir)
    
    print(f"[INFO] Found {len(labels_a_map)} label files in A")
    print(f"[INFO] Found {len(labels_b_map)} label files in B")
    print(f"[INFO] Found {len(images_map)} image files")

    # Only process images that have both labels AND the image file
    all_stems = sorted(set(labels_a_map.keys()) & set(labels_b_map.keys()) & set(images_map.keys()))
    print(f"[INFO] Images with both labels: {len(all_stems)}")

    stats_path = out_dir / "statistics.json"
    
    qualified_count = 0
    rejected_count = 0
    rejected_reasons: Dict[str, List[str]] = {}

    for stem in all_stems:
        # Get file paths
        label_a_path = labels_a_map[stem]
        label_b_path = labels_b_map[stem]
        img_path = images_map[stem]
        
        # Parse labels
        boxes_a = parse_yolo_label(label_a_path)
        boxes_b = parse_yolo_label(label_b_path)

        # Strict check: box count must be the same
        if len(boxes_a) != len(boxes_b):
            rejected_count += 1
            rejected_reasons[stem] = [f"Box count mismatch: A={len(boxes_a)}, B={len(boxes_b)}"]
            # Copy to rejected folder
            shutil.copy2(img_path, rejected_img_dir / img_path.name)
            shutil.copy2(label_a_path, rejected_lbl_a_dir / label_a_path.name)
            shutil.copy2(label_b_path, rejected_lbl_b_dir / label_b_path.name)
            continue

        # Match boxes (require same class)
        matches, un_a, un_b = match_boxes(
            boxes_a, boxes_b,
            require_same_class=True,
            min_iou_for_match=args.iou_threshold,
        )

        # Check if all boxes are matched
        if len(matches) != len(boxes_a):
            rejected_count += 1
            reasons = []
            if un_a:
                reasons.append(f"Unmatched boxes in A: {len(un_a)}")
            if un_b:
                reasons.append(f"Unmatched boxes in B: {len(un_b)}")
            # Check for class mismatches or low IoU
            for ia, ib, iou in matches:
                if boxes_a[ia].class_id != boxes_b[ib].class_id:
                    reasons.append(f"Class mismatch: box {ia} (A:{boxes_a[ia].class_id} vs B:{boxes_b[ib].class_id})")
                if iou < args.iou_threshold:
                    reasons.append(f"Low IoU: box {ia} (IoU={iou:.3f} < {args.iou_threshold})")
            rejected_reasons[stem] = reasons
            # Copy to rejected folder
            shutil.copy2(img_path, rejected_img_dir / img_path.name)
            shutil.copy2(label_a_path, rejected_lbl_a_dir / label_a_path.name)
            shutil.copy2(label_b_path, rejected_lbl_b_dir / label_b_path.name)
            continue

        # Check all matched boxes have same class and IoU >= threshold
        all_pass = True
        fail_reasons = []
        for ia, ib, iou in matches:
            if boxes_a[ia].class_id != boxes_b[ib].class_id:
                all_pass = False
                fail_reasons.append(f"Class mismatch: box {ia} (A:{boxes_a[ia].class_id} vs B:{boxes_b[ib].class_id})")
            if iou < args.iou_threshold:
                all_pass = False
                fail_reasons.append(f"Low IoU: box {ia} (IoU={iou:.3f} < {args.iou_threshold})")
        
        if not all_pass:
            rejected_count += 1
            rejected_reasons[stem] = fail_reasons
            # Copy to rejected folder
            shutil.copy2(img_path, rejected_img_dir / img_path.name)
            shutil.copy2(label_a_path, rejected_lbl_a_dir / label_a_path.name)
            shutil.copy2(label_b_path, rejected_lbl_b_dir / label_b_path.name)
            continue

        # QUALIFIED: randomly pick one annotation
        qualified_count += 1
        chosen_label = random.choice([label_a_path, label_b_path])
        shutil.copy2(img_path, qualified_img_dir / img_path.name)
        shutil.copy2(chosen_label, qualified_lbl_dir / chosen_label.name)

    # Write statistics JSON
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "labels_a": str(labels_a_dir),
                    "labels_b": str(labels_b_dir),
                    "images_dir": str(images_dir),
                    "iou_threshold": args.iou_threshold,
                },
                "stats": {
                    "total_images": len(all_stems),
                    "qualified_count": qualified_count,
                    "rejected_count": rejected_count,
                    "qualification_rate": round(qualified_count / max(1, len(all_stems)), 4),
                },
                "rejected_details": rejected_reasons,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"[OK] Statistics saved: {stats_path}")
    print(f"{'='*60}")
    print(f"Total images processed: {len(all_stems)}")
    print(f"Qualified (copied to dataset): {qualified_count}")
    print(f"Rejected (inconsistent): {rejected_count}")
    print(f"Qualification rate: {(qualified_count / max(1, len(all_stems))) * 100:.2f}%")
    print(f"\n[OUTPUT]")
    print(f"  Qualified dataset: {out_dir}")
    print(f"    - Images: {qualified_img_dir}")
    print(f"    - Labels: {qualified_lbl_dir}")
    print(f"  Rejected folder: {out_dir / 'rejected'}")
    print(f"    - Images: {rejected_img_dir}")
    print(f"    - Labels A: {rejected_lbl_a_dir}")
    print(f"    - Labels B: {rejected_lbl_b_dir}")


if __name__ == "__main__":
    main()
