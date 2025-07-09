# TimelapseFromPrompt
TimelapseFromPrompt is a tool for automatically generating object-centric timelapse videos using natural language prompts.

## About

🔍 What it does

- Input: A regular video recording of any scene (e.g., quadruped, animals, drone).

- Step 1 – Frame Extraction: Uses ffmpeg to extract frames from the video.

- Step 2 – Object Detection: Applies GroundingDINO to detect the object of interest based on a user-defined text prompt.

- Step 3 – Timelapse Creation: Selects and compiles relevant frames into a focused timelapse video of the specified object.

⚙️ Features

- 💬 Prompt-based detection – Just describe what you want to track (e.g., "drone", "person walking", "red ball").

- 🎞️ Frame interval control – Choose how often frames are sampled using frame_interval (default: 13).

- 📦 Detection thresholds – Tune BOX_THRESHOLD and TEXT_THRESHOLD for controlling detection sensitivity:

```python
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
```
🔁 Dynamic frame interval – Change frame interval mid-video to better highlight interesting segments.

## Installation 

TODO


## Troubleshooting

TODO

# ✒️ Citation

**TimelapseFromPrompt**

**Grounding dino**

```BibTeX
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```

