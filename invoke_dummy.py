"""Invoke the LipSync Model."""

from __future__ import annotations


import logging
import os
from random import randint
import shutil
from time import sleep
from typing import Literal
import cv2
from typing_extensions import TypedDict
from uuid import uuid4
import json
import datetime
import numpy as np


class Coordinates(TypedDict):
    top: int
    bottom: int
    left: int
    right: int


class FaceMask(TypedDict):
    size: float
    feathering: int
    mouth_tracking: bool
    debug_mask: bool


# type aliases
ModelQuality = Literal["Fast", "Improved", "Enhanced"]
ModelVersion = Literal["Wav2Lip", "Wav2Lip_GAN"]
OutputHeight = Literal["full resolution", "half resolution"] | int
Upscaler = Literal["gfpgan", "RestoreFormer"]


# defaults
DefaultFacePadding: Coordinates = {"top": 0, "bottom": 0, "left": 0, "right": 0}
DefaultFaceMask: FaceMask = {
    "size": 2.5,
    "feathering": 2,
    "mouth_tracking": False,
    "debug_mask": False,
}
DefaultBoundingBox: Coordinates = {"top": -1, "bottom": -1, "left": -1, "right": -1}
DefaultCrop: Coordinates = {"top": 0, "bottom": -1, "left": 0, "right": -1}

# constants
CWD = os.getcwd()
DATABASE_PATH = os.path.join(CWD, "database.json")
TEMP_PATH = os.path.join(CWD, "temp")
OUTPUT_PATH = os.path.join(CWD, "result")
MEL_STEP_SIZE = 16

os.makedirs(TEMP_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)


def save_config(
    run_id: str,
    data: dict,
) -> None:
    """Save the configuration to the database."""
    # create the database if it doesn't exist
    logger.info("Saving configuration to database")
    with open(DATABASE_PATH, "r") as file:
        logger.debug("Loading database")
        database = json.load(file)
        database = database if database else {}

    database[run_id] = data

    with open(DATABASE_PATH, "w") as file:
        logger.debug("Saving configuration")
        json.dump(database, file, indent=4)


def ensure_resouces(face: str, audio: str, output: str):
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)
    os.makedirs(TEMP_PATH, exist_ok=True)
    # remove last_detected_face.pkl if input face changes
    # get last entry in database
    if not os.path.exists(DATABASE_PATH):
        logger.debug("Creating database")
        with open(DATABASE_PATH, "w") as file:
            json.dump({}, file)
            database = {}
    else:
        with open(DATABASE_PATH, "r") as file:
            database = json.load(file)
    if database:
        last_entry = database[next(reversed(database))]
        last_face = last_entry["params"]["face"]
        if last_face != face:
            logger.info("Face changed, removing last_detected_face.pkl")
            os.remove(os.path.join(CWD, "last_detected_face.pkl"))


def validate_paths(face: str, audio: str, output: str):
    if not os.path.exists(face):
        raise RuntimeError(f"video/image file {face} not found")
    if os.path.isdir(face):
        raise RuntimeError(f"video/image file {face} is a directory")
    if not os.path.exists(audio):
        raise RuntimeError(f"audio file {audio} not found")
    if os.path.isdir(audio):
        raise RuntimeError(f"audio file {audio} is a directory")
    if os.path.exists(output):
        raise RuntimeError(f"output file {output} already exists")
    if os.path.isdir(output):
        raise RuntimeError(f"output file {output} is a directory")


def get_frames(
    face: str,
    fps: int,
    res_scale: int,
    output_height: int,
    rotate: bool,
    crop: tuple[int, int, int, int],
    is_image: bool,
) -> tuple[list[np.ndarray], float]:
    logger.info("Getting frames")
    if is_image:
        logger.debug("Image detected, returning single frame")
        return [cv2.imread(face)], float(fps)
    frames = []
    logger.debug("Reading video frames")
    stream = cv2.VideoCapture(face)
    _fps = stream.get(cv2.CAP_PROP_FPS)
    while True:
        reading, frame = stream.read()
        if not reading:
            stream.release()
            break
        if res_scale != 1:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(
                frame, (int(output_height / aspect_ratio), output_height)
            )
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        y1, y2, x1, x2 = crop
        if x2 == -1:
            x2 = frame.shape[1]
        if y2 == -1:
            y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        frames.append(frame)
    return frames, _fps


def main(
    face: str,
    audio: str,
    output: str | None = None,
    quality: ModelQuality = "Improved",
    version: ModelVersion = "Wav2Lip",
    height: OutputHeight = "full resolution",
    smooth: bool = False,
    padding: Coordinates = DefaultFacePadding,
    mask: FaceMask = DefaultFaceMask,
    bounding_box: Coordinates = DefaultBoundingBox,
    face_crop: Coordinates = DefaultCrop,
    rotate: bool = False,
    upscaler: Upscaler = "gfpgan",
    static: bool = False,
    frames_per_second: int = 25,
):
    """Invoke the LipSync Model.

    Args:
        face (str): Path to the video file containing the face.
        audio (str): Path to the audio file.
        output (str): Path to save the output video.
        quality (ModelQuality, optional): Quality of the model. Defaults to "Improved".
        version (ModelVersion, optional): Version of the model. Defaults to "Wav2Lip".
        smooth (bool, optional): Smooth the final video. Defaults to False.
        padding (Coordinates, optional): Padding around the face. Defaults to DefaultFacePadding.
        mask (FaceMask, optional): Masking options. Defaults to DefaultFaceMask.
        box (Coordinates, optional): Bounding box coordinates. Defaults to DefaultBoundingBox.
        crop (Coordinates, optional): Cropping coordinates. Defaults to DefaultCrop.
        rotate (bool, optional): Rotate the face. Defaults to False.
        super_resolution (bool, optional): Upscale the final video. Defaults to True.
        upscaler (Upscaler, optional): Upscaler to use. Defaults to "gfpgan".
        static (bool, optional): Use static image. Defaults to False.
        frames_per_second (int, optional): Frames per second of the final video. Defaults to 25.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = uuid4().hex[:8]
    face = os.path.abspath(face)
    audio = os.path.abspath(audio)
    if output is None:
        output = f"{run_id}.mp4"
    else:
        output = os.path.basename(output)
    if not output.endswith(".mp4"):
        output += ".mp4"
    output = os.path.abspath(os.path.join(OUTPUT_PATH, output))
    ensure_resouces(face, audio, output)
    data = locals()
    data.pop("run_id")
    save_config(
        run_id,
        {
            "timestamp": timestamp,
            "params": data,
        },
    )

    audio_folder, audio_file_with_ext = os.path.split(audio)
    audio_file, audio_ext = os.path.splitext(audio_file_with_ext)

    face_folder, face_file_with_ext = os.path.split(face)
    face_file, face_ext = os.path.splitext(face_file_with_ext)

    frames, fps = get_frames(
        face,
        frames_per_second,
        1,
        256,
        height == "full resolution",
        (face_crop["top"], face_crop["bottom"], face_crop["left"], face_crop["right"]),
        face_ext.lower() in (".jpg", ".jpeg", ".png"),
    )
    yield {"total": len(frames), "output": output}
    for i, frame in enumerate(frames, start=1):
        sleep(1 / randint(10, 100))
        yield {"progress": i}
