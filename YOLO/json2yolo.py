import os
import json
import glob

# 1. JSON 所在文件夹（你的 zip 里就是 train/）
JSON_DIR = "train"

# 2. 输出 txt 的文件夹（你可以改成自己习惯的，比如 labels/train）
OUT_DIR = "train_txt"
os.makedirs(OUT_DIR, exist_ok=True)

# 3. 类别映射表（目前你的数据里只有 "green cuboid"）
#   如果以后有别的，比如 "red cuboid"、"blue cuboid"，在这里加：
#   label2id = {"green cuboid": 0, "red cuboid": 1, "blue cuboid": 2}
label2id = {
    "green cuboid": 0,
    "purple cuboid": 1,
    "red cuboid": 2,
    "yellow cuboid": 3,
    "green triangle": 4,
    "orange cuboid": 5,  # 新增的类别
}


def convert_one_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    yolo_lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        points = shape["points"]  # [[x1, y1], [x2, y2]]
        (x1, y1), (x2, y2) = points

        # 有时候标的时候可能从右下到左上画，所以要取 min/max
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        # 计算中心点和宽高（像素）
        x_c = (x_min + x_max) / 2.0
        y_c = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min

        # 归一化到 0~1
        x_c_norm = x_c / img_w
        y_c_norm = y_c / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        class_id = label2id[label]

        line = f"{class_id} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        yolo_lines.append(line)

    # 输出 txt，文件名和 json 一样，只是后缀改成 .txt
    base = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(OUT_DIR, base + ".txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    print("saved:", out_path)


def main():
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    print("found", len(json_files), "json files")
    for jp in json_files:
        convert_one_json(jp)


if __name__ == "__main__":
    main()
