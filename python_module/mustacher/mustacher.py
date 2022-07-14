import sys
import pathlib
import math
import logging
import importlib.resources as resources

from . import resources as mustaches
from . import cascades
from . import imlog

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
    with resources.open_binary(mustaches, 'mustache.jpg') as mustache:
        mustache_data = mustache.read()

    return mustache_data

def mustache_image_data(image_data, mustache_data=get_default_mustache_data()):
    image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
    raw_result = mustache_raw_image(image, mustache_data)

    _, image_array = cv2.imencode('.JPEG', raw_result)
    image_array = np.array(image_array)

    return image_array.tobytes()

def mustache_raw_image(image, mustache_data=get_default_mustache_data()):
    mustache = cv2.imdecode(np.fromstring(mustache_data, np.uint8), cv2.IMREAD_UNCHANGED)

    return _mustache_image(image, mustache)

def _mustache_image(image, mustache):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detect(grayscale_image, FACES)
    eyes = detect(grayscale_image, EYES)
    eyes = get_eyes_for_face(faces[0], eyes)
    if len(eyes) == 2:
        angle = -1 * object_angle(eyes[0], eyes[1])
        mustache = rotate_image(mustache, angle)

    logging.info(f"Detected {len(faces)} faces")
    logging.info(f"Detected {len(eyes)} eyes")

    imlog.log_image_objects(image, [*eyes, *faces])

    for face in faces:
        image = put_mustache_on_face(image, face, mustache)

    return image

def get_eyes_for_face(face, eyes):
    """Determines which of the eyes belong to the given face"""
    eyes = [i for i in eyes if eye_within_face(face, i)]

    return eye_decider(face, eyes)

def eye_decider(face, eyes):
    """Decides which of the detected eyes are actually on that face"""
    if len(eyes) <= 2:
        return eyes

    return eyes[:2]

def eye_within_face(face, eye):
    face_x, face_y, face_w, face_h = face
    _, _, eye_w, eye_h = eye
    eye_x = eye[0] + int(0.5 * eye_w)
    eye_y = eye[1] + int(0.5 * eye_h)

    return face_x <= eye_x and face_y <= eye_y and face_x + face_w >= eye_x and face_y + face_h >= eye_y

def put_mustache_on_face(image, face, mustache):
    """ """
    face_cols, face_rows = (face[2], face[3])

    # Resize the mustache in proportion to the face size
    new_mustache_size = (int(face_cols / 2), int(face_rows / 2))
    mustache = cv2.resize(mustache, new_mustache_size, cv2.INTER_AREA)

    mustache_h, mustache_w = (mustache.shape[0], mustache.shape[1])

    # Find the center of the mustache overlay
    mustache_coords_center = (face[1] + int(.69 * face[3]), face[0] + int(.5 * face[2]))
    # Get the top left corner of the mustache overlay
    must_c, must_r = (
        mustache_coords_center[0] - int(.5 * mustache_h),
        mustache_coords_center[1] - int(0.5 * mustache_h)
    )

    im_width = image.shape[0]
    im_height = image.shape[1]

    left_pad = must_c - 1
    right_pad = im_width - must_c - mustache_w + 1
    top_pad = must_r - 1
    bottom_pad = im_height - must_r - mustache_h + 1

    mustache = np.pad(
        mustache,
        ((left_pad, right_pad), (top_pad, bottom_pad), (0, 0)),
        mode="constant", constant_values=255
    )
    # Invert the mustache (we want what is currently black to be visisble)
    mustache = cv2.bitwise_not(mustache)
    return cv2.subtract(image, mustache)

def object_angle(obj1, obj2):
    """Determines the angle between (the center point of) two objects"""
    c1 = (obj1[0] + int(obj1[2] / 2), obj1[1] + int(obj1[3] / 2))
    c2 = (obj2[0] + int(obj2[2] / 2), obj2[1] + int(obj2[3] / 2))
    dist = math.dist(c1, c2)
    return math.degrees(math.acos(abs(c1[0] - c2[0]) / dist))

def rotate_image(image, degrees):
    # This will rotate the image by "degrees" degrees using an affine transform
    height, width = image.shape[:2]
    M = cv2.getRotationMatrix2D(center=(height / 2, width / 2), angle=degrees, scale=1)
    image = cv2.warpAffine(image, M, image.shape[:2], borderValue=(255, 255, 255))
    return image

def detect(image, sheet):
    """Detect objects in an image based on the given cascade sheet name,
    the image must be a single layer (greyscale usually)"""
    with resources.path(cascades, sheet) as cascade_sheet_file:
        cascade_sheet = cv2.CascadeClassifier(str(cascade_sheet_file))
    objects = cascade_sheet.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
    return objects
