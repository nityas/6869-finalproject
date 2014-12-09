# based on http://cs229.stanford.edu/proj2013/JimenezNguyen_MathFormulas_final_paper.pdf

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/') # not necessary for all computers, depending on how OpenCV was installed

import cv2
import numpy as np
from skimage import morphology
from PIL import Image

fname = 'math.png'
img = cv2.imread(fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 10) # maybe increase the last argument
kernel = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(thresh, kernel, iterations=1)

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_color = cv2.imread(fname)
sizes = []
for cnt in contours:
    x, y, w, h, = cv2.boundingRect(cnt)
    sizes.append(w * h)
sizes.sort()

num = 0
for cnt in contours:
    x, y, w, h, = cv2.boundingRect(cnt)
    if w * h > sizes[len(sizes) * 3 / 4] / 2:
        cv2.imwrite('data/' + fname.replace('.', '_%d.' % num), img_color[y:y+h, x:x+w, :])
        num += 1

for cnt in contours:
    x, y, w, h, = cv2.boundingRect(cnt)
    if w * h > sizes[len(sizes) * 3 / 4] / 2:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("aaaa", img_color)
cv2.waitKey(0)
