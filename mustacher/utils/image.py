import enum
import math

import cv2

class OverlayImage:
    def __init__(self, opencv=cv2):
        self.opencv = opencv

    def call(self, overlay, base_image):
        if overlay.shape != base_image.shape:
            raise "Overlay and base image must be the same shape"
        # Invert the mustache (we want what is currently black to be visisble)
        overlay = self.opencv.bitwise_not(overlay)
        return self.opencv.subtract(base_image, overlay)

    def __call__(self, *args):
        return self.call(*args)

def object_angle(obj1, obj2):
    """Determines the angle between (the center point of) two DetectedObject's"""
    c1 = (obj1.x + int(obj1.width / 2), obj1.y + int(obj1.height / 2))
    c2 = (obj2.x + int(obj2.width / 2), obj2.y + int(obj2.height / 2))
    dist = math.dist(c1, c2)
    return math.degrees(math.acos(abs(c1[0] - c2[0]) / dist))

def rotate_image(image, degrees):
    # This will rotate the image by "degrees" degrees using an affine transform
    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D(center=(height / 2, width / 2), angle=degrees, scale=1)
    image = cv2.warpAffine(image, M, image.shape[:2], borderValue=(255, 255, 255))
    return image

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
