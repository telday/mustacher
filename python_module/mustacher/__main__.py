import argparse
import sys
import importlib.resources as resources

from .mustacher import mustache_image
from . import resources as mustaches

import cv2

parser = argparse.ArgumentParser(description='Add a mustache to faces on an image')
parser.add_argument('filename', help='The name of the input image')

args = parser.parse_args()

mustached = mustache_image(args.filename, output_filename='out.jpeg')
