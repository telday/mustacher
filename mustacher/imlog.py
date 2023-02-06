import os
import time
import os.path

import cv2
import numpy as np

LOGDIR = "logs"

def ensure_log_dir(wrapped):
    def between(*args):
        if not os.path.exists(LOGDIR):
            os.mkdir(LOGDIR)
        wrapped(*args)
    return between

@ensure_log_dir
def log_image(image):
    im_name = time.time()
    _, image_array = cv2.imencode('.JPEG', image)
    encoded_image = np.array(image_array)

    with open(f'{LOGDIR}/{im_name}.jpeg', 'wb') as file:
        file.write(encoded_image.tobytes())

@ensure_log_dir
def log_image_objects(image, objects):
    """Logs an image to a file with each detected object marked. Useful for debugging.
    Args:
        image (np.array): The image to log
        objects (List[DetectedObject]): The objects we want to mark
    """
    im_name = time.time()
    image = image.copy()

    for obj in objects:
        top_left = (obj.x, obj.y)
        bottom_right = (obj.x + obj.width, obj.y + obj.height)
        color = (255, 0, 0)
        image = cv2.rectangle(image, top_left, bottom_right, color, 5)

    _, image_array = cv2.imencode('.JPEG', image)
    encoded_image = np.array(image_array)

    with open(f'{LOGDIR}/{im_name}.jpeg', 'wb') as file:
        file.write(encoded_image.tobytes())
