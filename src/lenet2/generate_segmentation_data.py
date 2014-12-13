import gzip
import cPickle
from os import listdir
from PIL import Image, ImageChops
import numpy as np
import math

training, valid, blah = cPickle.load(gzip.open('EnglishHandwritten.gz', 'rb'))
output = 'SegTest.gz'

FINAL_IMAGE_SIZE = 28

def get_class(c):
    if c >= '0' and c <= '9':
        return ord(c) - ord('0')
    elif c >= 'A' and c <= 'Z':
        return ord(c) - ord('A') + 10
    else:
        return ord(c) - ord('a') + 36

images = []
ivalues = open('../../hog/data_2_answers.txt', 'r').readlines()
values = []
cur = 0
for i in range(0, 303):
    if len(ivalues[i].strip()) == 0:
        continue

    image = Image.open('../../hog/data_2/math_%d.png' % (i))
    image = image.convert('L') # convert to grayscale

    dim = image.size

    max_length = max(dim)

    size = (max_length, max_length)

    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size
    #print image_size

    thumb = image.crop( (0, 0, size[0], size[1]) )

    offset_x = max( (size[0] - image_size[0]) / 2, 0 )
    offset_y = max( (size[1] - image_size[1]) / 2, 0 )

    img = ImageChops.offset(thumb, offset_x, offset_y)
    img = img.resize((FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE), Image.ANTIALIAS)

    img = np.asarray(img, dtype='float64') / 255.0

    feature_vector = img.transpose().flatten()

    images.append(feature_vector)
    values.append(get_class(ivalues[i].strip()))

images = np.array(images)
values = np.array(values)

test = (images, values)

cPickle.dump((training, valid, test), gzip.open(output, 'wb'))
