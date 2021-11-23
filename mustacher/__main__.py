import cv2
import sys
from .mustacher import mustache_image
from . import resources as mustaches
import importlib.resources as resources


imagePath = sys.argv[1]

# read and convert image
image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
with resources.path(mustaches, 'mustache.jpg') as mustache_file:
    mustache = cv2.imread(str(mustache_file), cv2.IMREAD_UNCHANGED)

mustached = mustache_image(image, mustache)

cv2.imwrite('out.jpeg', mustached)
cv2.imshow("Faces found", mustached)
cv2.waitKey(0)
