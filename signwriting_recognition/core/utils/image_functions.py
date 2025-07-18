from core.utils.path_functions import create_directories_for_file
from pathlib import Path
import mediapipe as mp
import numpy as np
import cv2


def save_image_with_cv2(output_path: str, image: np.ndarray):
    """
    The function `save_image_with_cv2` saves an image to a specified output path after converting its
    color format from RGB to BGR using OpenCV.

    Args:
        output_path (str): The `output_path` parameter is a string that represents the file path where the
    image will be saved.
        image (np.ndarray): The `image` parameter is a NumPy array representing an image.
    """
    create_directories_for_file(output_path)
    cv2.imwrite(output_path, image)


def load_rgb(image_path: str | Path) -> mp.Image:
    """Lê qualquer imagem (RGB ou escala-de-cinza) e devolve sempre RGB."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)  # mantém 1 ou 3 canais
    if bgr is None:
        raise IOError(f"Erro ao ler {image_path}")

    if bgr.ndim == 2:                     # cinza → adiciona canais
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:               # RGBA → BGR
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)