#!/usr/bin/env python3
"""
RealSense + YOLO 实时检测 + 测距脚本

功能：
- 打开 RealSense 相机（彩色 + 深度，对齐到彩色）
- 使用你训练好的 YOLO 模型检测桌面上的积木（颜色+形状作为类别，比如 green cuboid）
- 对每个检测框，读取深度图中该框中心附近的深度，换算成米
- 在画面上实时显示：框 + 类别 + 置信度 + 距离（m）
- 按 q 退出
"""

import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ============ 根据你的实际情况修改这几个路径/参数 ============
MODEL_PATH = "/home/student06/runs/detect/train11/weights/best.pt"

# RealSense 分辨率（和你采集时保持一致更稳）
WIDTH, HEIGHT, FPS = 640, 480, 30

# YOLO 推理参数
CONF_THRES = 0.5   # 置信度阈值
IMG_SIZE = 640     # YOLO 输入尺寸
# ======================================================


def get_depth_distance(depth_img: np.ndarray, cx: int, cy: int, depth_scale: float) -> float | None:
    """
    在 depth 图中，以 (cx, cy) 为中心取一个 5x5 小窗口，取非零像素的中位数，乘 depth_scale 得到距离（米）。
    如果没有有效深度，返回 None。
    """
    h, w = depth_img.shape
    x1 = max(0, cx - 2)
    x2 = min(w, cx + 3)
    y1 = max(0, cy - 2)
    y2 = min(h, cy + 3)

    patch = depth_img[y1:y2, x1:x2]
    valid = patch[patch > 0]

    if valid.size == 0:
        return None

    depth_raw = float(np.median(valid))
    distance_m = depth_raw * depth_scale
    return distance_m


def main():
    # 1. 加载 YOLO 模型
    print(f"[INFO] 加载 YOLO 模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] 类别 names: {model.names}")

    # 2. 初始化 RealSense
    print("[INFO] 初始化 RealSense 管道...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)

    # 获取 depth scale（米/单位）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] depth_scale = {depth_scale} m/单位")

    # 对齐 depth 到 color
    align_to = rs.stream.color
    align = rs.align(align_to)

    print("[INFO] 实时检测开始，按 'q' 退出窗口。")

    try:
        while True:
            # 3. 读取并对齐一帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())   # HxWx3, BGR, uint8
            depth_image = np.asanyarray(depth_frame.get_data())   # HxW, uint16

            # 4. 用 YOLO 做检测
            results = model(color_image, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)[0]
            boxes = results.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    class_name = model.names.get(cls_id, str(cls_id))

                    # 计算中心点像素坐标
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # 5. 在深度图中读取距离
                    distance_m = get_depth_distance(depth_image, cx, cy, depth_scale)

                    # 6. 在画面上画框 + 文本
                    cv2.rectangle(
                        color_image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 0, 0),  # 蓝色框
                        2,
                    )

                    label = f"{class_name} {conf:.2f}"
                    if distance_m is not None:
                        label += f" {distance_m:.2f}m"
                    else:
                        label += " N/A"

                    txt_x, txt_y = int(x1), max(0, int(y1) - 5)
                    cv2.putText(
                        color_image,
                        label,
                        (txt_x, txt_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # 7. 显示结果
            cv2.imshow("RealSense YOLO Depth", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] 收到 'q'，退出。")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] RealSense 已停止。")


if __name__ == "__main__":
    main()
