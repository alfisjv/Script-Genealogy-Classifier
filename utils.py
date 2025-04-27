# Step 1: Setting up logger (utils.py)

import logging

logger = logging.getLogger("project_logger")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# utils.py

from PIL import Image
import numpy as np
import cv2
import torch


def skeletonize_image(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(img) > 127:
        img = 255 - img
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(binary, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()
        done = cv2.countNonZero(binary) == 0

    return skeleton


def compute_directional_features(img):
    img = np.array(img)
    skeleton = skeletonize_image(img)

    sobel_kernels = [
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
    ]

    channels = [cv2.filter2D(skeleton, -1, k) for k in sobel_kernels]
    stacked = np.stack(channels, axis=0)
    return torch.tensor(stacked / 255.0, dtype=torch.float32)
