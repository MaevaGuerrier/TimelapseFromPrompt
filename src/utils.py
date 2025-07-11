"""
utils for the main file
"""

import yaml
import ffmpeg
import torch
import shutil
import os
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from scipy.interpolate import splprep, splev
import numpy as np
import cv2


def get_config(config_path: str = None):
    """
    retrieve and send the config
    Args:
        config path
    returns
        yaml config
    """

    with open(config_path, encoding="utf-8") as stream:
        try:
            algo_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise FileNotFoundError("Error loading the configuration file") from exc

    return algo_config


def make_frames(input_video: str, frames_path: str, fps: int):
    """
    use ffmpeg to get the frames
    """
    output_pattern = os.path.join(frames_path, "%04d.png")
    # delete only if it exists
    shutil.rmtree(frames_path, ignore_errors=True)
    os.makedirs(frames_path, exist_ok=True)

    (ffmpeg.input(input_video).filter("fps", fps=fps).output(output_pattern).run())


def intersection_area(boxA, boxB):
    x1_a, y1_a, x2_a, y2_a = boxA
    x1_b, y1_b, x2_b, y2_b = boxB

    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)

    if x_right < x_left or y_bottom < y_top:
        return 0

    return (x_right - x_left) * (y_bottom - y_top)


def grabcut_mask_inside_box(img, box):
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2].copy()

    mask = np.zeros(crop.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (5, 5, crop.shape[1] - 10, crop.shape[0] - 10)  # slight inset

    cv2.grabCut(crop, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    crop[final_mask == 0] = 0  # remove background

    return crop


def get_grounding_output(
    model,
    image,
    caption,
    box_threshold,
    text_threshold=None,
    with_logits=True,
    cpu_only=False,
    token_spans=None,
):
    assert (
        text_threshold is not None or token_spans is not None
    ), "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption), token_span=token_spans
        ).to(
            image.device
        )  # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for token_span, logit_phr in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = " ".join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend(
                    [phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num]
                )
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


def get_spline(points):

    points = np.array(points)

    # Fit a spline
    tck, u = splprep(points.T, s=0)
    u_fine = np.linspace(0, 1, 100)
    x_fine, y_fine = splev(u_fine, tck)

    # Prepare points for OpenCV
    smooth_pts = np.array(list(zip(x_fine, y_fine)), dtype=np.int32)
    smooth_pts = smooth_pts.reshape((-1, 1, 2))
    return smooth_pts