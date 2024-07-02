"""Invoke the LipSync Model."""

from __future__ import annotations


import logging
import os
import shutil
import subprocess
from typing import Literal, cast
import cv2
import torch  # type: ignore[import]
from typing_extensions import TypedDict
from uuid import uuid4
import json
import datetime
from easy_functions import (
    get_input_length,
    get_video_details,
)
import contextlib
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip  # type: ignore[import]
import numpy as np
from audio import load_wav, melspectrogram
from enhance import load_sr, upscale
from inference import create_mask, create_tracked_mask, datagen, ns, do_load


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
logger.setLevel(logging.DEBUG)

def load_database() -> dict:
    """Load the database."""
    logger.info("Loading database")
    with open(DATABASE_PATH, "r") as file:
        database = json.load(file)
    return database or {}


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


def get_resolution_scale(height: OutputHeight, face: str, is_image: bool):
    logger.info("Getting resolution scale")
    if not is_image:
        _, in_height, _, _ = get_video_details(face)
    else:
        in_height = cv2.imread(face).shape[0]

    res_custom = False
    if height == "full resolution":
        scale = 1
    elif height == "half resolution":
        scale = 2
    else:
        scale = 3
        res_custom = True
    out_height = in_height // scale
    if res_custom:
        out_height = cast(int, height)
    logger.debug(f"Output height: {out_height}")
    logger.debug(f"Scale: {scale}")
    return out_height, scale


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


def get_mel_chunks(audio: str, fps: float):
    wav = load_wav(audio, 16000)
    mel = melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise RuntimeError(
            "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
        )
    mel_chunks = []

    mel_idx_multiplier = 80.0 / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + MEL_STEP_SIZE])
        i += 1
    return mel_chunks


def invoke_model(
    frames: list,
    mel_chunks: list,
    audio: str,
    output: str,
    fps: float,
    temp_output: str,
    mask: FaceMask,
    quality: ModelQuality,
):
    gen = datagen(frames.copy(), mel_chunks)
    for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
        if i == 0:
            if quality not in ("Fast", "Improved"):
                print(
                    f"mask size: {ns.mask_dilation}, feathering: {ns.mask_feathering}"
                )
                logger.info("Loading super resolution model")
                run_params = load_sr()
            logger.info("Starting inference...")
            frame_h, frame_w = frames[0].shape[:-1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            out = cv2.VideoWriter(
                temp_output,
                fourcc,
                fps,
                (frame_w, frame_h),
            )
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(
            ns.device
        )
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(
            ns.device
        )

        with torch.no_grad():
            pred = ns.model(mel_batch, img_batch)  # type: ignore

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c

            if mask["debug_mask"]:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            cf = f[y1:y2, x1:x2]

            if quality == "Enhanced":
                p = upscale(p, run_params)

            if quality in ("Improved", "Enhanced"):
                if mask["mouth_tracking"]:
                    p, _ = create_tracked_mask(p, cf)
                else:
                    p, _ = create_mask(p, cf)
            f[y1:y2, x1:x2] = p
            out.write(f)
        yield

    # Close the windows and release the video
    cv2.destroyAllWindows()
    out.release()

    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            temp_output,
            "-i",
            audio,
            "-c:v",
            "libx264",
            output,
        ]
    )


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
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data.pop("run_id")

    audio_folder, audio_file_with_ext = os.path.split(audio)
    audio_file, audio_ext = os.path.splitext(audio_file_with_ext)

    face_folder, face_file_with_ext = os.path.split(face)
    face_file, face_ext = os.path.splitext(face_file_with_ext)

    is_image = False

    if face_ext in [".jpg", ".jpeg", ".png"]:
        static = True
        is_image = True

    if version == "Wav2Lip_GAN":
        checkpoint_path = os.path.join(CWD, "checkpoints", "Wav2Lip_GAN.pth")
    else:
        checkpoint_path = os.path.join(CWD, "checkpoints", "Wav2Lip.pth")

    if mask["feathering"] == 3:
        mask["feathering"] = 5
    elif mask["feathering"] == 2:
        mask["feathering"] = 3

    output = os.path.join(OUTPUT_PATH, os.path.basename(output))
    validate_paths(face, audio, output)
    output_height, res_scale = get_resolution_scale(height, face, is_image)
    pad = (
        round(padding["top"] * res_scale),
        round(padding["bottom"] * res_scale),
        round(padding["left"] * res_scale),
        round(padding["right"] * res_scale),
    )
    box = (
        bounding_box["top"],
        bounding_box["bottom"],
        bounding_box["left"],
        bounding_box["right"],
    )
    crop = (
        face_crop["top"],
        face_crop["bottom"],
        face_crop["left"],
        face_crop["right"],
    )

    ns.nosmooth = not smooth
    ns.box = box
    ns.pads = pad
    ns.crop = crop
    ns.static = static
    ns.mask_dilation = mask["size"]
    ns.mask_feathering = mask["feathering"]
    ns.img_size = 96

    temp_output = os.path.join(TEMP_PATH, f"{run_id}_result.mp4")
    temp_face = os.path.join(TEMP_PATH, f"{run_id}_face{face_ext}")
    temp_audio = os.path.join(TEMP_PATH, f"{run_id}_audio.wav")

    shutil.copy(face, temp_face)

    if audio_ext == ".wav":
        shutil.copy(audio, temp_audio)
    else:
        # convert audio to wav
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                audio,
                temp_audio,
            ]
        )

    # trim video if it's longer than audio
    audio_length = get_input_length(audio)
    video_length = get_input_length(face) if not static else audio_length

    if video_length > audio_length:
        logger.info("Trimming video to match audio length")
        trimmed_video = os.path.join(TEMP_PATH, f"{run_id}_trimmed.mp4")
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            ffmpeg_extract_subclip(face, 0, audio_length, targetname=trimmed_video)
        temp_face = trimmed_video

    # subprocess.check_call(cmd)
    frames, fps = get_frames(
        temp_face, frames_per_second, res_scale, output_height, rotate, crop, is_image
    )
    mel_chunks = get_mel_chunks(temp_audio, fps)

    frames = frames[: len(mel_chunks)]
    print(str(len(frames)) + " frames to process")
    batch_size = 1
    ns.wav2lip_batch_size = batch_size
    total = int(np.ceil(len(mel_chunks) / batch_size))
    yield {"total": total, "output": output}
    frame_number = 1
    do_load(checkpoint_path)
    for _ in invoke_model(
        frames, mel_chunks, temp_audio, output, fps, temp_output, mask, quality
    ):
        yield {"progress": frame_number}
        frame_number += 1
    save_config(run_id, {"timestamp": timestamp, "params": data})
