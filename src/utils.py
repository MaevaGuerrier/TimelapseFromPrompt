"""
utils for the main file
"""

import yaml
import ffmpeg
import os


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
    output_pattern = os.path.join(frames_path, "frame_%04d.png")
    os.makedirs(frames_path, exist_ok=True)

    (ffmpeg.input(input_video).filter("fps", fps=fps).output(output_pattern).run())
