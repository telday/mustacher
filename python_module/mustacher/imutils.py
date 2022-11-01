import importlib.resources

from . import cascades

import cv2
import numpy as np

def pad_image(image, size, coords, value):
    """Pads an image with values value

    Args:
        image (numpy.ndarray): The image to pad
        size (tuple): The size of the final image (width, height)
        coords (tuple): The (x, y) coordinates of the top left pixel of the original image
            in the final padded image
        value (int): The value to use for padding
    """
    left_padding = coords[1] - 1
    right_padding = size[1] - coords[1] - image.shape[1] + 1
    top_padding = coords[0] - 1
    bottom_padding = size[0] - coords[0] - image.shape[0] + 1

    return cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, None, value)

def rotate_image(image, degrees):
    # This will rotate the image by "degrees" degrees using an affine transform
    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D(center=(height / 2, width / 2), angle=degrees, scale=1)
    image = cv2.warpAffine(image, M, image.shape[:2], borderValue=(255, 255, 255))
    return image

def detect(image, sheet):
    """Detect objects in an image based on the given cascade sheet name,
    the image must be a single layer (greyscale usually)"""
    with importlib.resources.path(cascades, sheet) as cascade_sheet_file:
        cascade_sheet = cv2.CascadeClassifier(str(cascade_sheet_file))

    objects = cascade_sheet.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
    return objects
