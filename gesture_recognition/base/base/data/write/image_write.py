from base.base.utilities import folder_utils
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
    folder_utils.create_directories_for_file(output_path)
    cv2.imwrite(output_path, image)
