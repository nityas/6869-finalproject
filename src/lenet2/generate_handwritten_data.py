import gzip
import cPickle
from os import listdir
from PIL import Image, ImageChops
import numpy as np
import math

output = 'EnglishNatural.gz'

FINAL_IMAGE_SIZE = 40

images = []
values = []
for group in listdir('data2/Bmp'):
    if '~' in group:
        continue
    print 'beginning group', group
    for png in listdir('data2/Bmp/%s' % group):
        image = Image.open('data2/Bmp/%s/%s' % (group, png))
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

        img = np.asarray(img, dtype='float64')

        feature_vector = img.flatten()

        images.append(feature_vector)
        values.append(int(group[len('Sample'):]) - 1)

    #break

images = np.array(images)
values = np.array(values)

print images.shape
print values.shape

# shuffle
perm = np.random.permutation(len(images))
ntraining = len(perm) / 2
nvalid = len(perm) / 4
training_idx = perm[:ntraining]
valid_idx = perm[ntraining:nvalid+ntraining]
test_idx = perm[ntraining+nvalid:]

training = (images[training_idx, :], values[training_idx])
valid = (images[valid_idx, :], values[valid_idx])
test = (images[test_idx, :], values[test_idx])

cPickle.dump((training, valid, test), gzip.open(output, 'wb'))
