import sys
import pathlib
import math
import logging
import importlib.resources

from . import resources as mustaches
from . import cascades
from . import imlog
from . import imutils

import cv2
import numpy as np

FACES = "cascade.xml"
EYES = "eye_cascade.xml"

def mustache_image(image_file, output_filename=None):
    with open(image_file, 'rb') as img:
        image_data = img.read()

    mustached_img_data = mustache_image_data(image_data)

    with open(output_filename, 'wb') as output:
        output.write(mustached_img_data)

    return mustached_img_data

def get_default_mustache_data():
    with importlib.resources.open_binary(mustaches, 'mustache.jpg') as mustache:
        mustache_data = mustache.read()

    return mustache_data

def get_default_mustache():
    image_data = get_default_mustache_data()
    return cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_UNCHANGED)

def mustache_image_data(image_data, mustache_data=get_default_mustache_data()):
    image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
    raw_result = mustache_raw_image(image, mustache_data)

    _, image_array = cv2.imencode('.JPEG', raw_result)
    image_array = np.array(image_array)

    return image_array.tobytes()

def mustache_raw_image(image, mustache_data=get_default_mustache_data()):
    mustache = cv2.imdecode(np.fromstring(mustache_data, np.uint8), cv2.IMREAD_UNCHANGED)

    return Image(image).image

class DetectedObject:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains(self, other):
        other_x, other_y = other.center()

        within_horizontal_bounds = self.x <= other_x and self.x + self.width >= other_x
        within_vertical_bounds = self.y <= other_y and self.y + self.height >= other_y

        return within_vertical_bounds and within_horizontal_bounds

    def center(self):
        return (self.x + int(0.5 * self.width), self.y + int(0.5 * self.height))

class Face(DetectedObject):
    def select_eyes(self, eyes):
        eyes = [i for i in eyes if self.contains(i)]
        if len(eyes) <= 2:
            self.eyes = eyes
        else:
            # Temporary way of deciding which 2 eyes to use
            self.eyes = eyes[:2]

    def upper_lip_coords(self):
        # Use standard face proportions to find the upper lip
        return (self.y + int(.69 * self.height), self.x + int(.5 * self.width))

    def angle(self):
        # If we didn't find 2 eyes for the given face assume it is vertical
        if len(self.eyes) < 2:
            return 0
        else:
            return object_angle(*self.eyes)

class Image:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

        self.__apply_cascades()
        for face in self.faces:
            face.select_eyes(self.eyes)
            self.image = self.place_mustache(face)

    def __apply_cascades(self):
        """"""
        grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.faces = [Face(*coords) for coords in imutils.detect(grayscale_image, FACES)]
        self.eyes = [DetectedObject(*coords) for coords in imutils.detect(grayscale_image, EYES)]

        logging.info(f"Detected {len(self.faces)} faces")
        logging.info(f"Detected {len(self.eyes)} eyes")

        imlog.log_image_objects(grayscale_image, [*self.faces, *self.eyes])

    def place_mustache(self, face):
        # Get the top left corner of the mustache overlay
        mustache = self.get_mustache(face)

        mustache_h, mustache_w = (mustache.shape[0], mustache.shape[1])
        must_c, must_r = (
            face.upper_lip_coords()[0] - int(0.5 * mustache_h),
            face.upper_lip_coords()[1] - int(0.5 * mustache_h)
        )

        mustache = imutils.pad_image(
            mustache, self.image.shape[:2], (must_c, must_r), (255, 255, 255)
        )

        # Invert the mustache (we want what is currently black to be visisble)
        mustache = cv2.bitwise_not(mustache)
        return cv2.subtract(self.image, mustache)

    def get_mustache(self, face):
        mustache = get_default_mustache()

        mustache = imutils.rotate_image(mustache, -1 * face.angle())
        new_mustache_size = (int(face.width / 2), int(face.height / 2))
        mustache = cv2.resize(mustache, new_mustache_size, cv2.INTER_AREA)

        return mustache

def object_angle(obj1, obj2):
    """Determines the angle between (the center point of) two detected objects"""
    c1 = (obj1.x + int(obj1.width / 2), obj1.y + int(obj1.height / 2))
    c2 = (obj2.x + int(obj2.width / 2), obj2.y + int(obj2.height / 2))
    dist = math.dist(c1, c2)
    return math.degrees(math.acos(abs(c1[0] - c2[0]) / dist))
