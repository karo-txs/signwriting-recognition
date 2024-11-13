import os
import numpy as np
import cv2


def gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


def brightness_contrast(image, brightness=0, contrast=30):
    # Ajuste de brilho e contraste
    return cv2.convertScaleAbs(image, alpha=1 + contrast / 127.0, beta=brightness)


def clahe(image):
    # Converte para o espaço de cores LAB e aplica CLAHE na luminância
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def resize_and_center_crop(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    center_h, center_w = h // 2, w // 2
    cropped_image = image[center_h - min_dim // 2:center_h + min_dim // 2,
                          center_w - min_dim // 2:center_w + min_dim // 2]
    return cv2.resize(cropped_image, target_size)


def adjust_gamma(image, gamma=0.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                     255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_transformations_and_save(image_path, output_path, transformations):
    image = cv2.imread(image_path)

    for transform in transformations:
        image = transform(image)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv2.imwrite(output_path, image)
