import argparse
import sys
import importlib.resources as resources

#from .mustacher import mustache_image, mustache_raw_image
from . import resources as mustaches
from .mustache_image import MustacheImage

import cv2
import numpy as np

def stream_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        print(frame.__class__, frame.shape)
        try:
            im = mustache_raw_image(frame)
        except Exception as e:
            print(e)
            im = frame
        # Display the resulting frame
        cv2.imshow('frame', im)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='Add a mustache to faces on an image')
parser.add_argument('-f', '--f', metavar='filename', help='The name of the input image')
parser.add_argument('--stream', help='Tells the program to actively get video', required=False, action='store_true')

args = parser.parse_args()
#
#if args.f:
#    mustached = mustache_image(args.f, output_filename='out.jpeg')
#
#if args.stream:
#    stream_video()
if args.f:
    with open(args.f, 'rb') as image_file:
        image_data = image_file.read()
    image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
    final_image = MustacheImage().call(image)
    cv2.imshow('frame', final_image)
