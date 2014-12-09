#CITATION: All data taken from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

import numpy
import sys
import os
from PIL import Image, ImageChops
import cPickle
import gzip
from skimage.feature import hog
import math

root = '../English/Img/'
bad = 'BadImag/Bmp/'
good = 'GoodImg/Bmp/'

HOG_TRAINING_DATA = 'data/hog_training_data'
HOG_TRAINING_LABELS = 'data/hog_training_labels'
HOG_TESTING_DATA = 'data/hog_testing_data'
HOG_TESTING_LABELS = 'data/hog_testing_labels'

IMG_TRAINING_DATA = 'data/img_training_data'
IMG_TRAINING_LABELS = 'data/img_training_labels'
IMG_TESTING_DATA = 'data/img_testing_data'
IMG_TESTING_LABELS = 'data/img_testing_labels'

IMG2D_TRAINING_DATA = 'data/img2d_training_data'
IMG2D_TRAINING_LABELS = 'data/img2d_training_labels'
IMG2D_TESTING_DATA = 'data/img2d_testing_data'
IMG2D_TESTING_LABELS = 'data/img2d_testing_labels'

TRAINING_DATA = ''
TRAINING_LABELS = ''
TESTING_DATA = ''
TESTING_LABELS = ''

#turn this flag on if we want to test on a harder data set
USE_BAD = False

# use this flag if we want to test on HOG feature vectors
HOG = False

#use this flag if we want to test on 1D feature vectors
ONE_D = True

training_data = []
training_labels = []
testing_data = []
testing_labels = []

NUM_CELLS = 144
FINAL_IMAGE_SIZE = 75

# Range endpoint should be 63, but leaving it as 2 for testing purposes.
for i in range (1, 63):
	print "Checking " + str(i)
	num = ''
	if i < 10:
		num = str(0) + str(i) + '/'
	else:
		num = str(i) + '/'

	if USE_BAD:
		sets = [bad, good]
	else:
		sets = [good]

	for data_set in sets:
		directory = root + data_set + "Sample0" + num
		files = os.listdir(directory)
		endpoints = []
		endpoints.append(len(files)/2)

		for j in range(len(files)):
			filename = directory + files[j]
			image = Image.open(open(filename))
			image = image.convert('L')

			# sketchy stuff from http://stackoverflow.com/questions/9103257/resize-image-maintaining-aspect-ratio-and-making-portrait-and-landscape-images-e
			dim = image.size

			max_length = max(dim)

			# 12 is sqrt(NUM_CELLS)
			cell_side = int(math.sqrt(NUM_CELLS))
			if (max(dim) % cell_side != 0):
				max_length = max(dim) + cell_side - (max(dim) % cell_side)

			size = (max_length, max_length)

			image.thumbnail(size, Image.ANTIALIAS)
			image_size = image.size
			#print image_size

			thumb = image.crop( (0, 0, size[0], size[1]) )

			offset_x = max( (size[0] - image_size[0]) / 2, 0 )
			offset_y = max( (size[1] - image_size[1]) / 2, 0 )

			img = ImageChops.offset(thumb, offset_x, offset_y)
			img = img.resize((FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE), Image.ANTIALIAS)

			img = numpy.asarray(img, dtype='float64')

			ppc = int(max_length / math.sqrt(NUM_CELLS))
			feature_vector = []

			if HOG:
				feature_vector = hog(img, orientations=8, pixels_per_cell=(ppc, ppc),
                    cells_per_block=(1, 1), visualise=False)
				print feature_vector
				TRAINING_DATA = HOG_TRAINING_DATA
				TRAINING_LABELS = HOG_TRAINING_LABELS
				TESTING_DATA = HOG_TESTING_DATA
				TESTING_LABELS = HOG_TESTING_LABELS

			elif ONE_D: 
				feature_vector = img.flatten()
				TRAINING_DATA = IMG_TRAINING_DATA
				TRAINING_LABELS = IMG_TRAINING_LABELS
				TESTING_DATA = IMG_TESTING_DATA
				TESTING_LABELS = IMG_TESTING_LABELS
			else:
				feature_vector = img
				print feature_vector.shape
				TRAINING_DATA = IMG2D_TRAINING_DATA
				TRAINING_LABELS = IMG2D_TRAINING_LABELS
				TESTING_DATA = IMG2D_TESTING_DATA
				TESTING_LABELS = IMG2D_TESTING_LABELS

			if j >= 0 and j < endpoints[0]:
				training_data.append(feature_vector)
				training_labels.append(i)
			elif j >= endpoints[0]:
				testing_data.append(feature_vector)
				testing_labels.append(i)


training_data = numpy.asarray(training_data)
training_labels = numpy.asarray(training_labels)
testing_data = numpy.asarray(testing_data)
testing_labels = numpy.asarray(testing_labels)

numpy.save(TRAINING_DATA, training_data)
numpy.save(TRAINING_LABELS, training_labels)
numpy.save(TESTING_DATA, testing_data)
numpy.save(TESTING_LABELS, testing_labels)