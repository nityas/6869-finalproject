# based on http://cs229.stanford.edu/proj2013/JimenezNguyen_MathFormulas_final_paper.pdf

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/') # not necessary for all computers, depending on how OpenCV was installed

import cv2
import numpy as np
from skimage import morphology

fname = 'math.png'
img_color = cv2.imread(fname)
img = cv2.imread(fname, 0)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 10)
kernel = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(thresh, kernel, iterations = 1)
output = dilation

contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h, = cv2.boundingRect(cnt)
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("aaaa", img_color)
cv2.waitKey(0)
