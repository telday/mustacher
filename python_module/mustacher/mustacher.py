import sys
import pathlib
import importlib.resources as resources

from . import resources as mustaches
from . import cascades

import cv2
import numpy as np

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
    mustache = cv2.imdecode(np.fromstring(mustache_data, np.uint8), cv2.IMREAD_UNCHANGED)

    image_mustachified = _mustache_image(image, mustache)

    _, image_array = cv2.imencode('.JPEG', image_mustachified)
    image_array = np.array(image_array)

    return image_array.tobytes()

def _mustache_image(image, mustache):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detect_faces(grayscale_image)

    for face in faces:
        image = put_mustache_on_face(image, face, mustache)

    return image

def put_mustache_on_face(image, face, mustache):
    """ """
    face_cols, face_rows = (face[2], face[3])

    # Resize the mustache in proportion to the face size
    new_mustache_size = (int(face_cols / 2), int(face_rows / 2))
    mustache = cv2.resize(mustache, new_mustache_size, cv2.INTER_AREA)
    mustache_h, mustache_w = (mustache.shape[0], mustache.shape[1])

    # Find the center of the mustache overlay
    mustache_coords_center = (face[1] + int(.69 * face[3]), face[0] + int(.5 * face[2]))
    must_c, must_r = (
        mustache_coords_center[0] - int(.5 * mustache_h),
        mustache_coords_center[1] - int(0.5 * mustache_h)
    )

    image_matrix = image[must_c:must_c + mustache_w, must_r:must_r + mustache_h]

    for col in range(len(mustache)):
        for row in range(len(mustache[col])):
            if sum(mustache[col][row]) <= 150:
                image_matrix[col][row] = mustache[col][row]

    overlay = image_matrix

    image[must_c:must_c + mustache_w, must_r:must_r + mustache_h] = overlay

    return image

def detect_faces(single_layer_image):
    """Detect faces in an image, the image must be a single layer (greyscale usually)"""
    with resources.path(cascades, "cascade.xml") as cascade_sheet_file:
        cascade_sheet = cv2.CascadeClassifier(str(cascade_sheet_file))

    return cascade_sheet.detectMultiScale(
            single_layer_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
