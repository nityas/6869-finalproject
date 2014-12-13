# based on http://cs229.stanford.edu/proj2013/JimenezNguyen_MathFormulas_final_paper.pdf

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/') # not necessary for all computers, depending on how OpenCV was installed

import cv2
import numpy as np
from skimage import morphology
from PIL import Image

def segment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2) # maybe increase the last argument
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.erode(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sizes = []
    for cnt in contours:
        x, y, w, h, = cv2.boundingRect(cnt)
        sizes.append(w * h)
    sizes.sort()

    res = []
    for cnt in contours:
        x, y, w, h, = cv2.boundingRect(cnt)
        if w * h > sizes[len(sizes) * 3 / 4] / 2:
            print cnt
            temp = np.copy(img[y:y+h, x:x+w, :])
            for i in range(h):
                for j in range(w):
                    if cv2.pointPolygonTest(cnt, (x + j, y + i), False) < 0:
                        temp[i, j, 0] = 0
                        temp[i, j, 1] = 0
                        temp[i, j, 2] = 0
                    else:
                        temp[i, j, 0] = 255
                        temp[i, j, 1] = 255
                        temp[i, j, 2] = 255
            res.append(temp)

    for cnt in contours:
        x, y, w, h, = cv2.boundingRect(cnt)
        if w * h > sizes[len(sizes) * 3 / 4] / 2:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.imshow("aaaa", img)
    #cv2.waitKey(0)

    return res

fname = 'captcha_1/0.1145622891645.jpg'
img = cv2.imread(fname)
res = segment(img)
for i in range(len(res)):
    print 'data_2/' + fname[10:].replace('.', '_%d.' % i)
    cv2.imwrite('data_2/' + fname[10:].replace('.', '_%d.' % i), res[i])
