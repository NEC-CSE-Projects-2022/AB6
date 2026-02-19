import cv2
import numpy as np
from PIL import Image


def is_leaf(image_path, green_threshold=0.30, edge_threshold=0.02):
    """
    Strong leaf validation using:
    - HSV green dominance
    - Edge/texture detection
    """

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception:
        return False

    # small resize for speed
    img = np.array(pil_img.resize((120, 120)))

    # =================================================
    # 1️⃣ GREEN CHECK (HSV)
    # =================================================
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    if green_ratio < green_threshold:
        return False

    # =================================================
    # 2️⃣ EDGE / TEXTURE CHECK
    # =================================================
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_ratio = np.sum(edges > 0) / edges.size

    if edge_ratio < edge_threshold:
        return False

    return True
