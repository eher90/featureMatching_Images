# Some introductory knowledge...
#-------------------------------

# We know that comparing the two images based on visual reference by feature matching yields little similarity and that is mostly based on similar small details between images.
# Also, OpenCV feature matching works only if you have two of the same images in different areas (think perspective, rotation, translation, etc.)

# My next idea is that if visual identification feature matching may not work, we can perhaps quantatiatively determine the simialrity based on the distribution of pixel colors.

import cv2
import matplotlib.pyplot as plt

image = cv2.imread('featureMatching_Images/images/artemisDrawing.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
plt.plot(histogram, color='k')
plt.show()