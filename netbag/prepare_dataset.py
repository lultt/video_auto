"""Convert Pascal VOC XML to YOLO format and split train/val."""
import os
import xml.etree.ElementTree as ET
import random
import shutil

ROOT = "J:/video_auto/netbag"
IMG_DIR = os.path.join(ROOT, "0515_first_net_raw")
XML_DIR = os.path.join(ROOT, "xml")
DATASET_DIR = os.path.join(ROOT, "dataset")

TRAIN_RATIO = 0.8
random.seed(42)

CLASS_MAP = {"0": 0}  # single class: netbag

def voc_to_yolo(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    labels = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        cls_id = CLASS_MAP.get(cls_name, 0)
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / float(w)
        bh = (ymax - ymin) / float(h)
        labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return labels


def main():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    xml_files = sorted([f for f in os.listdir(XML_DIR) if f.endswith(".xml")])
    random.shuffle(xml_files)

    split_idx = int(len(xml_files) * TRAIN_RATIO)
    splits = {"train": xml_files[:split_idx], "val": xml_files[split_idx:]}

    for split, files in splits.items():
        for xml_file in files:
            stem = os.path.splitext(xml_file)[0]
            img_file = stem + ".jpg"

            src_img = os.path.join(IMG_DIR, img_file)
            if not os.path.exists(src_img):
                print(f"WARNING: {src_img} not found, skipping")
                continue

            shutil.copy2(src_img, os.path.join(DATASET_DIR, "images", split, img_file))

            xml_path = os.path.join(XML_DIR, xml_file)
            labels = voc_to_yolo(xml_path)
            label_path = os.path.join(DATASET_DIR, "labels", split, stem + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(labels))

    print(f"Train: {len(splits['train'])} images")
    print(f"Val:   {len(splits['val'])} images")
    print("Dataset prepared at:", DATASET_DIR)


if __name__ == "__main__":
    main()
