import importlib

import cv2
import numpy as np

from mustacher import resources

class Configuration:
    @classmethod
    def debug(cls):
        return True

    @classmethod
    def mustache_bytes(cls):
        with importlib.resources.open_binary(resources, 'mustache.jpg') as mustache:
            mustache_data = mustache.read()

        return mustache_data

    @classmethod
    def mustache_array(cls):
        return cv2.imdecode(np.fromstring(cls.mustache_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
