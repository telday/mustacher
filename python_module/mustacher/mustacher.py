import cv2
import sys
import pathlib
import importlib.resources as resources

from . import resources as mustaches
from . import cascades

def mustache_image(image_file, output_filename=None):
    with resources.path(mustaches, 'mustache.jpg') as mustache_file:
        mustache = cv2.imread(str(mustache_file), cv2.IMREAD_COLOR)

    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

    image_mustachified = _mustache_image(image, mustache)

    if output_filename is not None:
        cv2.imwrite(output_filename, image_mustachified)

    return image_mustachified

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
