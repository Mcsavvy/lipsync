from types import SimpleNamespace
from typing import Literal
import torch
import numpy as np
from PIL import Image
import configparser
import math
import os
import pickle
import cv2
from batch_face import RetinaFace  # type: ignore[import]
from functools import partial
from tqdm import tqdm  # type: ignore[import]
from easy_functions import load_model, g_colab


class Namespace(SimpleNamespace):
    # Device options
    gpu_id: Literal[0, -1]
    device: Literal["cuda", "cpu", "mps"]

    #  options
    model: torch.nn.Module
    detector: RetinaFace
    detector_model: torch.nn.Module
    predictor: object
    mouth_detector: object

    # Config options
    mask_dilation: float
    mask_feathering: float
    pads: tuple[int, int, int, int]
    nosmooth: bool
    static: bool
    box: tuple[int, int, int, int]
    crop: tuple[int, int, int, int]
    img_size: int
    wav2lip_batch_size: int

    kernel: np.ndarray
    last_mask: np.ndarray
    x: int
    y: int
    w: int
    h: int

    mel_step_size: int


ns = Namespace(
    gpu_id=None,
    device=None,
    model=None,
    detector=None,
    detector_model=None,
    predictor=None,
    mouth_detector=None,
    mask_dilation=None,
    mask_feathering=None,
    pads=None,
    nosmooth=None,
    static=None,
    box=None,
    img_size=96,
    wav2lip_batch_size=None,
    kernel=None,
    last_mask=None,
    x=None,
    y=None,
    w=None,
    h=None,
    mel_step_size=16,
)

ns.device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
ns.gpu_id = 0 if torch.cuda.is_available() else -1

if ns.device == "cpu":
    print(
        "Warning: No GPU detected so inference will be done on the CPU which is VERY SLOW!"
    )


with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
    ns.predictor = pickle.load(f)

with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
    ns.mouth_detector = pickle.load(f)

# creating variables to prevent failing when a face isn't detected

in_g_colab = g_colab()

if not in_g_colab:
    # Load the config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Get the value of the "preview_window" variable
    preview_window = config.get("OPTIONS", "preview_window")

all_mouth_landmarks = []  # type: ignore


def do_load(checkpoint_path):
    ns.model = load_model(checkpoint_path)
    ns.detector = RetinaFace(
        gpu_id=ns.gpu_id, model_path="checkpoints/mobilenet.pth", network="mobilenet"
    )
    ns.detector_model = ns.detector.model


def face_rect(images):
    face_batch_size = 8
    num_batches = math.ceil(len(images) / face_batch_size)
    prev_ret = None
    for i in range(num_batches):
        batch = images[i * face_batch_size : (i + 1) * face_batch_size]
        all_faces = ns.detector(batch)  # return faces list of all images
        for faces in all_faces:
            if faces:
                box, landmarks, score = faces[0]
                prev_ret = tuple(map(int, box))
            yield prev_ret


def create_tracked_mask(img, original_img):
    # Convert color space from BGR to RGB if necessary
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    # Detect face
    faces = ns.mouth_detector(img)
    if len(faces) == 0:
        if ns.last_mask is not None:
            ns.last_mask = cv2.resize(ns.last_mask, (img.shape[1], img.shape[0]))
            mask = ns.last_mask  # use the last successful mask
        else:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
    else:
        face = faces[0]
        shape = ns.predictor(img, face)

        # Get points for mouth
        mouth_points = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
        )

        # Calculate bounding box dimensions
        ns.x, ns.y, ns.w, ns.h = cv2.boundingRect(mouth_points)

        # Set kernel size as a fraction of bounding box size
        kernel_size = int(max(ns.w, ns.h) * ns.mask_dilation)
        # if kernel_size % 2 == 0:  # Ensure kernel size is odd
        # kernel_size += 1

        # Create kernel
        ns.kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Create binary mask for mouth
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)

        ns.last_mask = mask  # Update last_mask with the new mask

    # Dilate the mask
    dilated_mask = cv2.dilate(mask, ns.kernel)

    # Calculate distance transform of dilated mask
    dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

    # Normalize distance transform
    cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

    # Convert normalized distance transform to binary mask and convert it to uint8
    _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
    masked_diff = masked_diff.astype(np.uint8)

    # make sure blur is an odd number
    blur = ns.mask_feathering
    if blur % 2 == 0:
        blur += 1
    # Set blur size as a fraction of bounding box size
    blur = int(max(ns.w, ns.h) * blur)  # 10% of bounding box size
    if blur % 2 == 0:  # Ensure blur size is odd
        blur += 1
    masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

    # Convert numpy arrays to PIL Images
    input1 = Image.fromarray(img)
    input2 = Image.fromarray(original_img)

    # Convert mask to single channel where pixel values are from the alpha channel of the current mask
    mask = Image.fromarray(masked_diff)

    # Ensure images are the same size
    assert input1.size == input2.size == mask.size

    # Paste input1 onto input2 using the mask
    input2.paste(input1, (0, 0), mask)

    # Convert the final PIL Image back to a numpy array
    input2 = np.array(input2)

    # input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
    cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)

    return input2, mask


def create_mask(img, original_img):
    # Convert color space from BGR to RGB if necessary
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    if ns.last_mask is not None:
        ns.last_mask = np.array(ns.last_mask)  # Convert PIL Image to numpy array
        ns.last_mask = cv2.resize(ns.last_mask, (img.shape[1], img.shape[0]))
        mask = ns.last_mask  # use the last successful mask
        mask = Image.fromarray(mask)

    else:
        # Detect face
        faces = ns.mouth_detector(img)
        if len(faces) == 0:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
        else:
            face = faces[0]
            shape = ns.predictor(img, face)

            # Get points for mouth
            mouth_points = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
            )

            # Calculate bounding box dimensions
            ns.x, ns.y, ns.w, ns.h = cv2.boundingRect(mouth_points)

            # Set kernel size as a fraction of bounding box size
            kernel_size = int(max(ns.w, ns.h) * ns.mask_dilation)
            # if kernel_size % 2 == 0:  # Ensure kernel size is odd
            # kernel_size += 1

            # Create kernel
            ns.kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Create binary mask for mouth
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)

            # Dilate the mask
            dilated_mask = cv2.dilate(mask, ns.kernel)

            # Calculate distance transform of dilated mask
            dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

            # Normalize distance transform
            cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

            # Convert normalized distance transform to binary mask and convert it to uint8
            _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
            masked_diff = masked_diff.astype(np.uint8)

            if not ns.mask_feathering == 0:
                blur = ns.mask_feathering
                # Set blur size as a fraction of bounding box size
                blur = int(max(ns.w, ns.h) * blur)  # 10% of bounding box size
                if blur % 2 == 0:  # Ensure blur size is odd
                    blur += 1
                masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

            # Convert mask to single channel where pixel values are from the alpha channel of the current mask
            mask = Image.fromarray(masked_diff)

            ns.last_mask = mask  # Update last_mask with the final mask after dilation and feathering

    # Convert numpy arrays to PIL Images
    input1 = Image.fromarray(img)
    input2 = Image.fromarray(original_img)

    # Resize mask to match image size
    # mask = Image.fromarray(mask)
    mask = mask.resize(input1.size)

    # Ensure images are the same size
    assert input1.size == input2.size == mask.size

    # Paste input1 onto input2 using the mask
    input2.paste(input1, (0, 0), mask)

    # Convert the final PIL Image back to a numpy array
    input2 = np.array(input2)

    # input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
    cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)

    return input2, mask


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T :]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, results_file="last_detected_face.pkl"):
    # If results file exists, load it and return
    if os.path.exists(results_file):
        print("Using face detection data from last input")
        with open(results_file, "rb") as f:
            return pickle.load(f)

    results = []
    pady1, pady2, padx1, padx2 = ns.pads

    tqdm_partial = partial(tqdm, position=0, leave=True)
    for image, (rect) in tqdm_partial(
        zip(images, face_rect(images)),
        total=len(images),
        desc="detecting face in every frame",
        ncols=100,
    ):
        if rect is None:
            cv2.imwrite(
                "temp/faulty_frame.jpg", image
            )  # check this frame where the face was not detected.
            raise ValueError(
                "Face not detected! Ensure the video contains a face in all the frames."
            )

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not ns.nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [
        [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
        for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]

    # Save results to file
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    return results


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    print("\r" + " " * 100, end="\r")
    if ns.box[0] == -1:
        if not ns.static:
            face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print("Using the specified bounding box instead of face detection...")
        y1, y2, x1, x2 = ns.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if ns.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        try:
            face, coords = face_det_results[idx].copy()
        except IndexError:
            print("Out of range!")
            print("Index: ", idx)
            print("Length of frames: ", len(frames))
            print("Length of face_det_results: ", len(face_det_results))
            raise

        face = cv2.resize(face, (ns.img_size, ns.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= ns.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, ns.img_size // 2 :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, ns.img_size // 2 :] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        yield img_batch, mel_batch, frame_batch, coords_batch