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
    im_name = time.time()
    image = image.copy()

    for obj in objects:
        top_left = (obj[0], obj[1])
        bottom_right = (obj[0] + obj[2], obj[1] + obj[3])
        color = (255, 0, 0)
        image = cv2.rectangle(image, top_left, bottom_right, color, 5)

    _, image_array = cv2.imencode('.JPEG', image)
    encoded_image = np.array(image_array)

    with open(f'{LOGDIR}/{im_name}.jpeg', 'wb') as file:
        file.write(encoded_image.tobytes())
