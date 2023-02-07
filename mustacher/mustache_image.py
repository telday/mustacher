import logging
import functools

import numpy as np

from mustacher.config import Configuration
from mustacher.utils.image import OverlayImage
import mustacher.utils.image
import mustacher.utils.detect_object
import mustacher.imlog

def derive_face_angle(face: np.ndarray):
    """Takes in an image which is cropped down to just a detected face and
    attempts to find the angle of the given face.

    Args:
        face (np.ndarray)
    Returns: (float) The calulated angle in radians
    """
    eyes = mustacher.utils.detect_object.detect_object(
        face.object_image,
        mustacher.utils.detect_object.CascadeSheets.EYE.value
    )
    # TODO rework this algorithm to better determine which of the
    # detected eyes are real if there are more than 2 detected
    if len(eyes) < 2:
        # We assume a 0 angle if we can't find 2 or more eyes
        return 0

    angle = mustacher.utils.image.object_angle(*eyes[:2])
    logging.debug(f"Calculated object angle: {angle}")
    return angle

class MustacheFace:
    def __init__(self, overlay_image=OverlayImage()):
        self.overlay_image = overlay_image

    def call(self, face: mustacher.utils.detect_object.DetectedObject):
        face_angle = derive_face_angle(face)
        mustache = Configuration.mustache_array()
        mustache = mustacher.utils.image.rotate_image(mustache, -1 * face_angle)

        mustacher.imlog.log_image(mustache)

        mustache_h, mustache_w = (mustache.shape[0], mustache.shape[1])
        upper_lip_coords = (int(.69 * face.height), int(.5 * face.width))
        must_c, must_r = (
            upper_lip_coords[0] - int(0.5 * mustache_h),
            upper_lip_coords[1] - int(0.5 * mustache_h)
        )

        # Mustache and face image must be the same size because of how we
        # overlay the images
        mustache = mustacher.utils.image.pad_image(
            mustache, face.object_image.shape[:2], (must_c, must_r), (255, 255, 255)
        )


        result = self.overlay_image(mustache, face.object_image)
        mustacher.imlog.log_image(result)
        return result

    def __call__(self, *args):
        return self.call(*args)

class MustacheImage:
    def __init__(self,
        mustache_face=MustacheFace()
     ):
        self.mustache_face = mustache_face

    def call(self, image):
        faces = mustacher.utils.detect_object.detect_object(
            image,
            mustacher.utils.detect_object.CascadeSheets.FACE.value
        )
        for face in faces:
            mustached_face = self.mustache_face(face)
            image[face.y:face.y + face.width, face.x:face.x + face.height] = mustached_face

        return image
