# RealSense + YOLO Block Detection

This repository provides a small end-to-end pipeline for detecting colored blocks on a table using an **Intel RealSense** RGB-D camera and a **YOLO** model.

The pipeline covers:

1. Collecting RGB + depth images with basic metadata
2. Converting LabelMe JSON annotations into YOLO format
3. Running real-time object detection with distance estimation from the depth map

------------------------------------------------------------------------------------------------------------------------------------------

## Scripts Overview

### 1. `capture_realsense.py` – RGB + Depth Data Collection

This script captures synchronized RGB and depth frames from an Intel RealSense camera and saves them to disk, along with simple labels provided via the terminal.

**Main features:**

* Starts a RealSense pipeline with aligned color and depth streams
* Displays a live preview window:

  * Left: RGB image
  * Right: depth visualization (grayscale)
* Key actions:

  * `s`

    * Save RGB image as `.jpg` to `dataset/images/`
    * Save raw depth frame as `.npy` to `dataset/depth/`
    * Ask in the terminal for:

      * `color` (e.g. green, purple, red, yellow, …)
      * `shape` (e.g. cuboid, triangle)
      * optional `notes`
    * Append a row to `dataset/labels.csv` with
      `filename, depthfile, color, shape, notes`
    * Create or update `dataset/metadata.json` with `depth_scale` (meters per depth unit)
  * `q`

    * Exit the program

**Output structure example:**

dataset/
  images/
    20250101_123000_000000.jpg
    …
  depth/
    20250101_123000_000000.npy
    …
  labels.csv
  metadata.json

---------------------------------------------------------------------------------------------------------

### 2. `json2yolo.py` – LabelMe JSON → YOLO TXT Conversion

This script converts LabelMe-style JSON annotations into YOLO TXT format.

**Config:**

* `JSON_DIR = "train"` – folder where `.json` annotation files are stored
* `OUT_DIR = "train_txt"` – folder where YOLO `.txt` files will be written

**Class mapping (`label2id`):**

label2id = {
  "green cuboid": 0,
  "purple cuboid": 1,
  "red cuboid": 2,
  "yellow cuboid": 3,
  "green triangle": 4,
  "orange cuboid": 5,
}

Each LabelMe JSON file is expected to contain:

* `imageWidth`
* `imageHeight`
* `shapes`: a list where each shape has

  * `label` – must match one of the keys in `label2id`
  * `points` – two points defining the bounding box: `[[x1, y1], [x2, y2]]`

**The script will:**

1. Read each JSON file in `JSON_DIR`
2. For every shape:

   * Build a bounding box from the two points
   * Compute box center `(x_c, y_c)` and size `(w, h)` in pixels
   * Normalize `x_c, y_c, w, h` to `[0, 1]`
3. Write one `.txt` file per image in YOLO format:

`<class_id> <x_center> <y_center> <width> <height>`

**Run:**

python json2yolo.py

Output files are saved to `train_txt/` with the same base filename as the JSON.

--------------------------------------------------------------------------------------------------------------------------------

### 3. `realsense_yolo_depth.py` – Real-Time Detection + Distance

This script performs real-time object detection on RGB frames and estimates distance using the aligned depth map.

**Main workflow:**

1. Load a trained YOLO model, e.g.

   MODEL_PATH = "/path/to/your/best.pt"

2. Initialize the RealSense pipeline:

   * Color stream: `WIDTH x HEIGHT` (e.g. 640 x 480) at `FPS` (e.g. 30)
   * Depth stream: same resolution and FPS
   * Align depth to color
   * Read `depth_scale` from the depth sensor

3. For each frame:

   * Get aligned color and depth images
   * Run YOLO inference on the color image
   * For each detection:

     * Get bounding box `(x1, y1, x2, y2)`
     * Compute box center `(cx, cy)`
     * Extract a 5×5 patch around `(cx, cy)` from the depth image
     * Filter out zero depth values, take the median of remaining pixels
     * Multiply by `depth_scale` to get distance in meters
   * Draw:

     * A bounding box around the object
     * A label above the box with:

       * class name (from `model.names`)
       * confidence
       * estimated distance in meters (e.g. `0.57 m`, or `N/A` if no valid depth)

4. Display the annotated RGB image in a window

5. Press `q` to quit and stop the RealSense pipeline

-----------------------------------------------------------------------------------------------------------------------------

## Requirements

* Python 3.8+
* Intel RealSense RGB-D camera (e.g. D435 / D435i)
* YOLO (Ultralytics YOLOv8 or compatible)

Install core Python packages:

pip install opencv-python numpy ultralytics pyrealsense2

Optional (for LabelMe annotations):

pip install labelme

Installing `pyrealsense2` depends on your OS; please refer to the official Intel RealSense SDK guide.

------------------------------------------------------------------------------------------------------------

## Setup

1. Clone or download this repository.

2. (Optional) Create and activate a virtual environment:

   * Linux/macOS:
     python -m venv venv
     source venv/bin/activate

   * Windows:
     python -m venv venv
     venv\Scripts\activate

3. Install dependencies:

   pip install -r requirements.txt

   or install the packages listed above manually.

4. Edit `MODEL_PATH` in `realsense_yolo_depth.py` to point to your YOLO weights, for example:

   MODEL_PATH = "/absolute/or/relative/path/to/best.pt"

-----------------------------------------------------------------------------------------------------

## Usage

### 1. Collect RGB + Depth Data

Run:

python capture_realsense.py

* `s` – save one frame:

  * RGB image to `dataset/images/*.jpg`
  * depth array to `dataset/depth/*.npy`
  * append label info to `dataset/labels.csv`
* `q` – exit

--------------------------------------------------------------------------------------------------

### 2. Convert LabelMe JSON to YOLO Format

Organize your LabelMe annotations under `train/`, e.g.:

train/
  img_0001.json
  img_0002.json
  ...

Then run:

python json2yolo.py

YOLO label files will be generated in:

train_txt/
  img_0001.txt
  img_0002.txt
  ...

You can then move these `.txt` files into your YOLO dataset structure, such as `labels/train` and `labels/val`.

------------------------------------------------------------------------------

### 3. Run Real-Time YOLO + Depth Detection

Before running, make sure:

* Your YOLO model is trained on the same class list as `label2id`
* `MODEL_PATH` in `realsense_yolo_depth.py` is correctly set

Run:

python realsense_yolo_depth.py

You should see:

* A live RGB window with bounding boxes
* Each box labeled with class, confidence, and estimated distance in meters
* Press `q` to exit
