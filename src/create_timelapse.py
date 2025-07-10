"""
Main script for creating a timelapse
"""

import os
from pathlib import Path
import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
from utils import get_config, make_frames
import supervision as sv
import yaml
import ffmpeg
import argparse

if __name__ == "__main__":

    # === Paths ===
    file_path = Path(__file__).resolve().parent
    # print(f"Using base directory: {basedir}")

    # Load Dino model
    model = load_model(
        f"{file_path}/config_dino.py",
        f"{file_path}/../weights/groundingdino_swint_ogc.pth",
    )

    # Get the config files from
    configs = get_config(f"{file_path}/../config.yaml")
    print(configs)

    for idx, config in enumerate(configs):
        print(f"Processing config {idx + 1}/{len(configs)}: {config}")
        algo = config
        video_path = f"{file_path}/../{configs[config]['video_dir']}"
        frames_path = f"{file_path}/../frames/{algo}/"
        print(video_path)
        output_path = f"{algo}.png"
        fps = configs[config]["fps"]
        make_frames(input_video=video_path, frames_path=frames_path, fps=fps)

        # === Load Grounding DINO model ===

        TEXT_PROMPT = configs[config]["prompt"]
        BOX_THRESHOLD = configs[config]["box_threshold"]
        TEXT_THRESHOLD = configs[config]["text_threshold"]
        frame_interval = configs[config]["frame_interval_begin"]  # 8 wwas limo

        # === Frame list ===
        frame_files = sorted(
            [
                f
                for f in os.listdir(frames_path)
                if f.endswith(".png") or f.endswith(".jpg")
            ],
            reverse=False,
        )

        # === Load first frame as background using OpenCV ===
        background_path = os.path.join(frames_path, frame_files[0])
        background_bgr = cv2.imread(background_path)
        robot_overlay_discrete = background_bgr.copy()

        total_images = len(frame_files)

        change_index = int(2 * total_images // 3)  # final third

        # === Process each frame ===
        for idx, filename in enumerate(frame_files):
            # increase frame interval near the end
            if idx == change_index:
                frame_interval = configs[config]["frame_interval_end"]
            if idx % frame_interval != 0:
                continue

            img_path = os.path.join(frames_path, filename)

            # Load with OpenCV
            frame_bgr = cv2.imread(img_path)

            # Convert BGR to RGB for Grounding DINO
            frame_raw, frame = load_image(img_path)

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
