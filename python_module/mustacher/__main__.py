import cv2
import sys
from .mustacher import mustache_image
from . import resources as mustaches
import importlib.resources as resources


imagePath = sys.argv[1]

mustached = mustache_image(imagePath, output_filename='out.jpeg')
