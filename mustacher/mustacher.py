import cv2
import sys
import pathlib
import importlib.resources as resources

from . import cascades

def get_cascade_classifier():
    pass

def mustache_image(image, mustache):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #cascade_sheet_file = pathlib.Path(str(cascades.__path__)) / 'cascades.xml'
    with resources.path(cascades, "cascade.xml") as cascade_sheet_file:
        cascade_sheet = cv2.CascadeClassifier(str(cascade_sheet_file))

    # detect faces in the image
    faces = cascade_sheet.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

    '''
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    '''

    for face in faces:
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
                if sum(mustache[col][row]) <= 80:
                    image_matrix[col][row] = mustache[col][row]

        overlay = image_matrix

        image[must_c:must_c + mustache_w, must_r:must_r + mustache_h] = overlay

    return image
