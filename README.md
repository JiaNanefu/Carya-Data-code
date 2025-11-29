# Carya Dataset Utilities

This repository contains a collection of standalone Python scripts for preparing and analyzing the **Carya** object detection dataset.  
They help you:

- Convert between annotation formats (YOLO / VOC / COCO)
- Create train/val/test splits with stratification
- Analyze class distributions
- Build rich metadata tables for further analysis

> **Note:** Most scripts contain **hard−coded Windows paths** pointing to the original dataset.  
> Before running them on your own data, **modify the paths at the top of each script**.

---

## Environment

- Python 3.x
- Recommended packages:
  - `numpy`
  - `opencv-python` (`cv2`)
  - `pandas`
  - `tqdm`
  - `scikit-learn`
  - `Pillow`

Install (example):

```bash
pip install numpy opencv-python pandas tqdm scikit-learn pillow
```

---

## Scripts Overview

### 1. `generate_carya_metadata.py`

Builds detailed **image-level** and **instance-level** metadata from images and YOLO labels.

- Reads images and corresponding YOLO `.txt` files.
- Computes:
  - Per-image attributes: width, height, number of fruits, counts per maturity level, mean image brightness, etc.
  - Per-instance attributes: class id, maturity label (`maturity1/2/3`, `blurring`), normalized bbox, pixel bbox, bbox area, ROI brightness, etc.
- Returns metadata as `pandas.DataFrame` objects that can be saved for analysis (e.g., to CSV or Parquet).

**Typical use:**

```bash
python generate_carya_metadata.py
```

Edit the image/label root directories and any output saving logic inside the script before running.

---

### 2. `labels_analyze.py`

Counts target objects per class in a label folder to understand **class distribution**.

Supports two modes:

- **YOLO mode (`mode = "yolo"`):**
  - Reads `.txt` label files.
  - Counts occurrences of each class ID (first column in each line).

- **VOC mode (`mode = "voc"`):**
  - Reads PASCAL VOC `.xml` files.
  - Counts occurrences of each object `name` tag.

Prints:

- Per-class counts
- Total number of bounding boxes
- Number of distinct classes

**Typical use:**

```bash
python labels_analyze.py
```

Set `mode`, and adjust the `label_dir` / `xml_dir` paths in `__main__`.

---

### 3. `stratified_split.py`

Performs a **stratified train/val/test split** for an object detection dataset that has a **rare class**.

- Inputs:
  - A folder of images (`SOURCE_IMAGE_DIR`)
  - A folder of labels (YOLO `.txt` or VOC `.xml`, controlled by `LABEL_EXTENSION`)
- Logic:
  - Scans all labels and separates images into:
    - A **rare pool**: images containing `RAREST_CLASS_ID`
    - A **common pool**: images without this class
  - Uses `train_test_split` twice to create:
    - Train / Val / Test splits for both pools
  - Merges pools so that each split has a controlled presence of the rare class.
- Outputs:
  - Copies images and labels into:
    - `OUTPUT_DIR/train/images` and `OUTPUT_DIR/train/labels`
    - `OUTPUT_DIR/val/images` and `OUTPUT_DIR/val/labels`
    - `OUTPUT_DIR/test/images` and `OUTPUT_DIR/test/labels`

**Typical use:**

```bash
python stratified_split.py
```

Before running, configure:

- `SOURCE_IMAGE_DIR`, `SOURCE_LABEL_DIR`
- `OUTPUT_DIR`
- `IMAGE_EXTENSION`, `LABEL_EXTENSION`
- `RAREST_CLASS_ID`, `SPLIT_RATIOS`, `RANDOM_SEED`

---

### 4. `voc_to_yolo.py`

Converts annotations from **PASCAL VOC XML** format to **YOLO TXT** format.

- Reads `.xml` files from `VOC_LABEL_DIR`.
- For each object:
  - Maps class name to an integer ID via `CLASS_NAME_TO_ID`.
  - Converts absolute VOC bbox (`xmin, ymin, xmax, ymax`) to normalized YOLO format:
    - `class_id x_center y_center width height` (all normalized to `[0, 1]`).
- Saves `.txt` files to `YOLO_OUTPUT_DIR`.

**Typical use:**

```bash
python voc_to_yolo.py
```

Edit:

- `VOC_LABEL_DIR`
- `YOLO_OUTPUT_DIR`
- `CLASS_NAME_TO_ID` (class mapping)  
  to match your dataset.

---

### 5. `yolo_to_voc.py`

Converts annotations from **YOLO TXT** format to **PASCAL VOC XML** format.

- Reads images from `IMAGE_DIR` and YOLO labels from `LABEL_DIR`.
- Uses `CLASS_ID_TO_NAME` to map integer class IDs to class names.
- Converts normalized YOLO bboxes back to absolute VOC bboxes.
- Writes standard VOC `annotation` XML files to `VOC_OUTPUT_DIR`.

**Typical use:**

```bash
python yolo_to_voc.py
```

Configure at the top:

- `IMAGE_DIR`
- `LABEL_DIR`
- `VOC_OUTPUT_DIR`
- `CLASS_ID_TO_NAME`

---

### 6. `yolo_to_coco.py`

Converts a **YOLO-formatted dataset with splits** into **COCO JSON** annotations.

Assumes dataset structure:

```text
dataset_root/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

Features:

- For each split (`train`, `val`, `test` by default):
  - Loads class names from `train/labels/classes.txt` if present, otherwise falls back to a default.
  - Reads images (`.jpg`, `.png`) and corresponding YOLO label files.
  - Builds `images`, `annotations`, `categories` sections following the COCO format.
  - Saves `instances_<split>.json` into `dataset_root/coco_annotations/`.

**Typical use:**

```bash
python yolo_to_coco.py
```

By default it uses the directory of the script as `dataset_root`.  
Adjust paths or arguments in `__main__` if needed.

---

### 7. `yolo_to_coco_single.py`

Converts a **single YOLO folder** (all images + labels) into **one COCO JSON file**.

- Inputs:
  - `IMAGES_DIR`: folder with all images.
  - `LABELS_DIR`: folder with YOLO `.txt` labels.
  - `OUTPUT_JSON`: path for output COCO JSON.
  - `CLASS_NAMES`: list of class names ordered by class ID.
- Supports multiple image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, both lower/upper case).
- Validates class IDs and bounding boxes, clips boxes within image bounds.
- Writes a single `annotations.json` in COCO format and prints a detailed summary.

**Typical use:**

```bash
python yolo_to_coco_single.py
```

Edit the constants at the top of the file (`IMAGES_DIR`, `LABELS_DIR`, `OUTPUT_JSON`, `CLASS_NAMES`) for your dataset.

---

## How to Use This Repository

1. **Clone to your machine** and open in your IDE.
2. For each script you plan to use:
   - **Update the hard-coded paths** and class mappings to fit your dataset.
   - Optionally adjust parameters (e.g., split ratios, rare class ID).
3. Run the scripts from the command line:

```bash
python <script_name>.py
```

4. Use the generated splits, annotations, and metadata in your training/analysis pipelines.

This README is designed for direct use on GitHub. You can save it as `README.md` in the root of this `code` folder or at the repository root.
