import pyrealsense2 as rs
import numpy as np
import cv2
import os
import csv
import json
from datetime import datetime

OUT_DIR = "dataset"
IMG_DIR = os.path.join(OUT_DIR, "images")
DEPTH_DIR = os.path.join(OUT_DIR, "depth")
CSV_PATH = os.path.join(OUT_DIR, "labels.csv")
META_PATH = os.path.join(OUT_DIR, "metadata.json")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

# RealSense pipeline setup
pipeline = rs.pipeline()
cfg = rs.config()
# 可改分辨率：640x480 或 1280x720
WIDTH, HEIGHT = 640, 480
cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

profile = pipeline.start(cfg)
# 获取 depth scale
dev = profile.get_device()
depth_sensor = dev.first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # meters per depth unit

# 保存 metadata（depth_scale）
meta = {"depth_scale_m_per_unit": float(depth_scale)}
with open(META_PATH, 'w') as f:
    json.dump(meta, f, indent=2)

# 对齐 depth 到 color
align = rs.align(rs.stream.color)

# 初始化 CSV
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'depthfile', 'color', 'shape', 'notes'])

print("REALSENSE capture running. 按 's' 保存一帧 (会要求输入 color, shape)。按 'q' 退出。")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())  # BGR uint8
        depth_image = np.asanyarray(depth_frame.get_data())  # uint16

        # 可视化 depth（仅用于显示）
        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)

        # 合并显示
        combined = np.hstack((color_image, cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)))
        cv2.putText(combined, "Press 's' to save, 'q' to quit", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        cv2.imshow('realsense_capture', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_name = f"{ts}.jpg"
            depth_name = f"{ts}.npy"
            img_path = os.path.join(IMG_DIR, img_name)
            depth_path = os.path.join(DEPTH_DIR, depth_name)

            # save rgb and raw depth array (uint16). depth units -> meters via depth_scale in metadata
            cv2.imwrite(img_path, color_image)
            np.save(depth_path, depth_image)

            # get label inputs
            color_label = input("color label (e.g. red): ").strip()
            shape_label = input("shape label (e.g. cuboid): ").strip()
            notes = input("notes (optional): ").strip()

            # write csv
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([img_name, depth_name, color_label, shape_label, notes])

            print(f"Saved: {img_name}, {depth_name}")
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Stopped.")