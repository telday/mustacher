import argparse
import sys
import importlib.resources as resources

from .mustacher import mustache_image, mustache_raw_image
from . import resources as mustaches

import cv2

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

if args.f:
    mustached = mustache_image(args.f, output_filename='out.jpeg')

if args.stream:
    stream_video()
