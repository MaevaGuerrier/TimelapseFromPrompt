"""
Main script for creating a timelapse
"""

import os
from pathlib import Path
import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
from utils import (
    get_config,
    make_frames,
    get_grounding_output,
    intersection_area,
    grabcut_mask_inside_box,
    get_spline,
)
import supervision as sv
import numpy as np
import yaml
import ffmpeg
import argparse

# from segment_anything import sam_model_registry, SamPredictor

# sam_checkpoint = "/home/somz/projects/TimelapseFromPrompt/weights/sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# device = "cpu"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# predictor = SamPredictor(sam)

if __name__ == "__main__":

    # === Paths ===
    file_path = Path(__file__).resolve().parent
    # print(f"Using base directory: {basedir}")

    # Load Dino model
    model = load_model(
        f"{file_path}/config_dino.py",
        f"{file_path}/../weights/groundingdino_swint_ogc.pth",
        device="cpu",
    )

    # Get the config files from
    configs = get_config(f"{file_path}/../config.yaml")
    print(configs)

    for idx, config in enumerate(configs):
        print(f"Processing config {idx + 1}/{len(configs)}: {config}")
        algo = config
        video_path = f"{file_path}/../{configs[config]['video_dir']}"
        frames_path = f"{file_path}/../frames/{algo}/"
        output_path = f"{algo}.png"
        fps = configs[config]["fps"]
        token_spans = configs[config]["token_spans"]
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

        print(frame_files)

        # === Load first frame as background using OpenCV ===
        background_path = os.path.join(frames_path, frame_files[0])
        background_bgr = cv2.imread(background_path)
        robot_overlay_discrete = background_bgr.copy()
        points = []
        total_images = len(frame_files)
        prev_box = None
        change_index = int(2 * total_images // 3)  # final third

        # === Process each frame ===
        for idx, filename in enumerate(frame_files):
            # increase frame interval near the end
            if idx == change_index:
                frame_interval = configs[config]["frame_interval_end"]
            if idx % frame_interval != 0 and idx != total_images - 1:
                continue

            img_path = os.path.join(frames_path, filename)

            # Load with OpenCV
            frame_bgr = cv2.imread(img_path)

            # Convert BGR to RGB for Grounding DINO
            frame_raw, frame = load_image(img_path)

            # Grounding DINO expects RGB (0-255)
            # boxes, logits, phrases = predict(
            #     model=model,
            #     image=frame,
            #     caption=TEXT_PROMPT,
            #     box_threshold=BOX_THRESHOLD,
            #     text_threshold=TEXT_THRESHOLD,
            # )

            boxes_filt, pred_phrases = get_grounding_output(
                model=model,
                image=frame,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                cpu_only=False,
                token_spans=eval(f"{token_spans}"),
            )

            H, W, _ = frame_bgr.shape

            if boxes_filt.shape[0] != 0:

                print(idx, boxes_filt, pred_phrases)
                cx, cy, w, h = boxes_filt[0].tolist()
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

                iou = 0
                if prev_box is not None:
                    area_a = (x1 - x0) * (y1 - y0)
                    x0_p, y0_p, x1_p, y1_p = prev_box
                    area_b = (x1_p - x0_p) * (y1_p - y0_p)
                    iou = intersection_area((x0, y0, x1, y1), prev_box) / (
                        area_a + area_b - intersection_area((x0, y0, x1, y1), prev_box)
                    )

                print(iou)
                if iou > 0.05:
                    print("very near old detection")
                    continue

                print("patching")
                # annotated_frame = frame_bgr.copy()
                # cv2.rectangle(annotated_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # annotated_frame = annotate(
                #     image_source=frame_bgr, boxes=boxes_filt, logits=None, phrases=None
                # )
                # annotated_frame = annotate(image_source=frame_bgr, boxes=boxes, logits=logits, phrases=phrases)

                # sv.plot_image(annotated_frame, (16, 16))
                # # show image
                # cv2.imshow(f"Annotated Frame {idx}", annotated_frame)
                # cv2.waitKey(300)  # Allow OpenCV to update the window
                # cv2.destroyAllWindows()

                # Crop from the current frame and paste onto the overlay
                # overlay an arrow

                patch = frame_bgr[y0:y1, x0:x1]

                # input_box = boxes_filt[0].cpu().numpy()
                # predictor.set_image(frame_bgr)
                # masks, _, _ = predictor.predict(
                #     point_coords=None,
                #     point_labels=None,
                #     box=input_box[None, :],
                #     multimask_output=False,
                # )

                # # Apply mask to image (keep only the masked area, rest black
                # masked_image = cv2.bitwise_and(frame_bgr, frame_bgr, mask=masks[0])
                # cv2.imshow("remove background", masked_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows
                # patch = grabcut_mask_inside_box(img=frame_bgr, box=(x0, y0, x1, y1))
                if patch.shape[0] > 0 and patch.shape[1] > 0:
                    robot_overlay_discrete[y0:y1, x0:x1] = patch

                points.append((int((x0 + x1) / 2), int(y1 - 100)))

                prev_box = (x0, y0, x1, y1)

        print(points)
        print(H, W)
        spline_points = get_spline(points)
        print(spline_points.shape)

        # cv2.polylines(
        #     robot_overlay_discrete,
        #     points,
        #     isClosed=False,
        #     color=(255, 0, 0),
        #     thickness=2,
        # )
        cv2.polylines(
            robot_overlay_discrete,
            [spline_points],
            isClosed=False,
            color=(255, 0, 0),
            thickness=2,
        )

        cv2.imwrite("debug.png", background_bgr)

        # === Save result ===
        cv2.imwrite(output_path, robot_overlay_discrete)
        print(f"✅ Saved overlay result to: {output_path}")
