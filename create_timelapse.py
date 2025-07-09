import os
import cv2
import numpy as np
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
from pathlib import Path
import supervision as sv
import matplotlib.pyplot as plt

# === Paths ===
basedir = Path(__file__).resolve().parent

# print(f"Using base directory: {basedir}")

algo = "sac_cbf_drone"
frames_dir = f"{basedir}/{algo}/"
output_path = f"{algo}.png"

# === Load Grounding DINO model ===
model = load_model(
    f"{basedir}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    f"{basedir}/GroundingDINO/weights/groundingdino_swint_ogc.pth",
)

TEXT_PROMPT = "caged drone"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
frame_interval = 13 # 8 wwas limo

# === Frame list ===
frame_files = sorted(
    [f for f in os.listdir(frames_dir) if f.endswith(".png") or f.endswith(".jpg")],
    reverse=False,
)

# === Load first frame as background using OpenCV ===
background_path = os.path.join(frames_dir, frame_files[0])
background_bgr = cv2.imread(background_path)
robot_overlay_discrete = background_bgr.copy()

total_images = len(frame_files)


change_index = int(2 * total_images // 3)  # final third


# === Process each frame ===
for idx, filename in enumerate(frame_files):
    # increase frame interval near the end 
    if idx == change_index:
        frame_interval = 20
    if idx % frame_interval != 0:
        continue

    frame_path = os.path.join(frames_dir, filename)

    # Load with OpenCV
    frame_bgr = cv2.imread(frame_path)

    # Convert BGR to RGB for Grounding DINO
    frame_raw, frame = load_image(frame_path)

    # Grounding DINO expects RGB (0-255)
    boxes, logits, phrases = predict(
        model=model,
        image=frame,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    H, W, _ = frame_bgr.shape


    # annotated_frame = annotate(image_source=frame_bgr, boxes=boxes, logits=logits, phrases=phrases)
    



    # sv.plot_image(annotated_frame, (16, 16))
    # # show image
    # # cv2.imshow(f"Annotated Frame {idx}", annotated_frame)
    # # cv2.waitKey(10)  # Allow OpenCV to update the window
    # # cv2.destroyAllWindows()

    if boxes.shape[0] > 0:
        max_idx = torch.argmax(logits).item()
        cx, cy, w, h = boxes[max_idx].tolist()
        # Convert from center format to top-left/bottom-right
        x0 = int((cx - w / 2) * W)
        y0 = int((cy - h / 2) * H)
        x1 = int((cx + w / 2) * W)
        y1 = int((cy + h / 2) * H)

        # Clamp to valid pixel range
        x0 = max(0, min(x0, W - 1))
        x1 = max(0, min(x1, W - 1))
        y0 = max(0, min(y0, H - 1))
        y1 = max(0, min(y1, H - 1))

        # Crop from the current frame and paste onto the overlay
        patch = frame_bgr[y0:y1, x0:x1]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            robot_overlay_discrete[y0:y1, x0:x1] = patch






# === Save result ===
cv2.imwrite(output_path, robot_overlay_discrete)
print(f"✅ Saved overlay result to: {output_path}")
