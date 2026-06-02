"""Extract reference frame at 21:18:51 and generate ROI design images."""
import os
import cv2
import numpy as np

OUT_DIR = "J:/video_auto/roi_design"
os.makedirs(OUT_DIR, exist_ok=True)

VIDEO = "C:/video/0515/ch01_20250515_205339_214053topspeed.mp4"
WALL_START = 20 * 3600 + 53 * 60 + 39  # 205339
TARGET = 21 * 3600 + 18 * 60 + 51       # 211851

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps
wall_end = 21 * 3600 + 40 * 60 + 53
wall_duration = wall_end - WALL_START
speed_ratio = wall_duration / video_duration

offset_sec = (TARGET - WALL_START) / speed_ratio
frame_idx = int(offset_sec * fps)

print(f"Video: {os.path.basename(VIDEO)}")
print(f"FPS: {fps:.2f}, Speed ratio: {speed_ratio:.3f}")
print(f"Target: 21:18:51, Frame index: {frame_idx}")

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Failed to read frame")
    exit(1)

h, w = frame.shape[:2]
print(f"Frame size: {w}x{h}")

# Output 1: Raw reference
raw_path = os.path.join(OUT_DIR, "roi_reference_raw.jpg")
cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"1. {raw_path}")

# Output 2: 100px grid
grid100 = frame.copy()
color_line = (0, 255, 255)  # yellow
color_text = (0, 0, 255)    # red
font = cv2.FONT_HERSHEY_SIMPLEX

for x in range(0, w + 1, 100):
    cv2.line(grid100, (x, 0), (x, h), color_line, 1)
for y in range(0, h + 1, 100):
    cv2.line(grid100, (0, y), (w, y), color_line, 1)

# Label key coordinates
labels = [(0, 0), (500, 300), (1000, 600), (1500, 900), (2000, 1200)]
for lx, ly in labels:
    cv2.putText(grid100, f"({lx},{ly})", (lx + 5, ly + 20), font, 0.5, color_text, 1, cv2.LINE_AA)
    cv2.circle(grid100, (lx, ly), 4, color_text, -1)

# Corners
cv2.putText(grid100, "(0,0)", (5, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(grid100, f"({w},0)", (w - 130, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(grid100, f"(0,{h})", (5, h - 10), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(grid100, f"({w},{h})", (w - 160, h - 10), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

grid100_path = os.path.join(OUT_DIR, "roi_reference_grid_100.jpg")
cv2.imwrite(grid100_path, grid100, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"2. {grid100_path}")

# Output 3: 50px grid
grid50 = frame.copy()
color_line_50 = (128, 255, 128)  # light green

for x in range(0, w + 1, 50):
    thickness = 1 if x % 100 == 0 else 1
    c = color_line if x % 100 == 0 else color_line_50
    cv2.line(grid50, (x, 0), (x, h), c, thickness)
for y in range(0, h + 1, 50):
    c = color_line if y % 100 == 0 else color_line_50
    cv2.line(grid50, (0, y), (w, y), c, 1)

# Label every 200px for readability
for x in range(0, w + 1, 200):
    cv2.putText(grid50, str(x), (x + 2, 15), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
for y in range(0, h + 1, 200):
    cv2.putText(grid50, str(y), (2, y + 12), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

# Corners
cv2.putText(grid50, "(0,0)", (5, 25), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(grid50, f"({w},0)", (w - 120, 25), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(grid50, f"(0,{h})", (5, h - 10), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(grid50, f"({w},{h})", (w - 150, h - 10), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

grid50_path = os.path.join(OUT_DIR, "roi_reference_grid_50.jpg")
cv2.imwrite(grid50_path, grid50, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"3. {grid50_path}")

# Output 4: Blank ROI label image
roi_blank = frame.copy()
overlay = roi_blank.copy()
font_roi = cv2.FONT_HERSHEY_SIMPLEX

# Place 3 label boxes at approximate zones (just labels, no regions)
zone_labels = [
    ("NET_ZONE", (w // 2, h // 3)),
    ("UNLOAD_ZONE", (w // 4, 2 * h // 3)),
    ("TRANSFER_ZONE", (3 * w // 4, 2 * h // 3)),
]

for label, (cx, cy) in zone_labels:
    text_size = cv2.getTextSize(label, font_roi, 1.0, 2)[0]
    bx1 = cx - text_size[0] // 2 - 10
    by1 = cy - text_size[1] // 2 - 10
    bx2 = cx + text_size[0] // 2 + 10
    by2 = cy + text_size[1] // 2 + 10
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.7, roi_blank, 0.3, 0, roi_blank)
    overlay = roi_blank.copy()
    cv2.rectangle(roi_blank, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
    cv2.putText(roi_blank, label, (bx1 + 10, by2 - 10), font_roi, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

roi_blank_path = os.path.join(OUT_DIR, "roi_reference_blank_roi.jpg")
cv2.imwrite(roi_blank_path, roi_blank, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"4. {roi_blank_path}")

print("\nDone.")
