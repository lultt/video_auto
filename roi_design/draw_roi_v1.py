"""Draw ROI zones on reference frame for visual confirmation."""
import cv2
import numpy as np

IMG = "J:/video_auto/roi_design/roi_reference_raw.jpg"
OUT1 = "J:/video_auto/roi_design/roi_layout_v1.jpg"
OUT2 = "J:/video_auto/roi_design/roi_layout_v1_with_coords.jpg"

ZONES = [
    ("LEFT", (500, 900, 1500, 1440), (0, 0, 255)),       # red
    ("TRANSFER", (950, 550, 1600, 1440), (0, 255, 0)),   # green
    ("RIGHT", (1500, 900, 2560, 1440), (255, 0, 0)),     # blue
]

font = cv2.FONT_HERSHEY_SIMPLEX

# Output 1: zones with labels
img1 = cv2.imread(IMG)
for name, (x1, y1, x2, y2), color in ZONES:
    cv2.rectangle(img1, (x1, y1), (x2, y2), color, 3)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    text_size = cv2.getTextSize(name, font, 1.2, 3)[0]
    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(img1, name, (tx, ty), font, 1.2, color, 3, cv2.LINE_AA)

cv2.imwrite(OUT1, img1, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"1. {OUT1}")

# Output 2: zones with corner coordinates
img2 = cv2.imread(IMG)
for name, (x1, y1, x2, y2), color in ZONES:
    cv2.rectangle(img2, (x1, y1), (x2, y2), color, 3)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    text_size = cv2.getTextSize(name, font, 1.2, 3)[0]
    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(img2, name, (tx, ty), font, 1.2, color, 3, cv2.LINE_AA)

    # Corner labels
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    offsets = [(5, -8), (-100, -8), (5, 20), (-100, 20)]
    for (cx_, cy_), (ox, oy) in zip(corners, offsets):
        label = f"({cx_},{cy_})"
        cv2.circle(img2, (cx_, cy_), 5, color, -1)
        cv2.putText(img2, label, (cx_ + ox, cy_ + oy), font, 0.5, color, 2, cv2.LINE_AA)

cv2.imwrite(OUT2, img2, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"2. {OUT2}")
print("Done.")
