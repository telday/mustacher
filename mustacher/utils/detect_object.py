import enum
import importlib

import cv2

from mustacher import imlog
from mustacher import cascades

class CascadeSheets(enum.Enum):
    def __EYE():
        with importlib.resources.path(cascades, "eye_cascade.xml") as cascade:
            return cv2.CascadeClassifier(str(cascade))

    def __FACE():
        with importlib.resources.path(cascades, "cascade.xml") as face_cascade:
            return cv2.CascadeClassifier(str(face_cascade))

    EYE = __EYE()
    FACE = __FACE()

class DetectedObject:
    """Represents an object detected in an image
    Args:
        x (int): The x coord of the top left corner of the detected object
        y (int): The y coord of the top left corner of the detected object
        width (int): Width of the object in px
        height (int): The height of the object in px
        image (numpy.ndarray): The image the object was detected in
    """
    def __init__(self, x, y, width, height, image):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.object_image = image[self.y:self.y + self.width, self.x:self.x + self.height]

        imlog.log_image(self.object_image)

    def contains(self, other):
        other_x, other_y = other.center()

        within_horizontal_bounds = self.x <= other_x and self.x + self.width >= other_x
        within_vertical_bounds = self.y <= other_y and self.y + self.height >= other_y

        return within_vertical_bounds and within_horizontal_bounds

    def center(self):
        return (self.x + int(0.5 * self.width), self.y + int(0.5 * self.height))

def detect_object(image, cascade_sheet):
    objects = cascade_sheet.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
    detected_objects = [DetectedObject(*obj, image) for obj in objects]
    imlog.log_image_objects(image, detected_objects)
    return detected_objects
