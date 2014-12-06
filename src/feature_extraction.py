#CITATION: All data taken from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

import numpy
import sys
import os
from PIL import Image
import cPickle
import gzip
from skimage.feature import hog

root = '../English/Img/'
bad = 'BadImag/Bmp/'
good = 'GoodImg/Bmp/'

# HOG_TRAINING_DATA = open('data/hog_training_data.csv', 'w+')
# HOG_TRAINING_LABELS = open('data/hog_training_labels.csv', "w+")
# HOG_TESTING_DATA = open('data/hog_testing_data.csv', "w+")
# HOG_TESTING_LABELS = open('data/hog_testing_labels.csv', "w+")

HOG_TRAINING_DATA = 'data/hog_training_data'
HOG_TRAINING_LABELS = 'data/hog_training_labels'
HOG_TESTING_DATA = 'data/hog_testing_data'
HOG_TESTING_LABELS = 'data/hog_testing_labels'

#turn this flag on if we want to test on a harder data set
USE_BAD = False

training_data = []
training_labels = []
testing_data = []
testing_labels = []

# Range endpoint should be 63, but leaving it as 2 for testing purposes.
for i in range (1, 2):
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
			img = Image.open(open(filename))

			img = img.convert('L')
			img = numpy.asarray(img, dtype='float64')

			feature_vector = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=False)

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

numpy.save(HOG_TRAINING_DATA, training_data)
numpy.save(HOG_TRAINING_LABELS, training_labels)
numpy.save(HOG_TESTING_DATA, testing_data)
numpy.save(HOG_TESTING_LABELS, testing_labels)

#print training_data
# numpy.savetxt(HOG_TRAINING_DATA, training_data, delimiter=",")
# numpy.savetxt(HOG_TRAINING_LABELS, training_labels, fmt='%i', delimiter=",")
# numpy.savetxt(HOG_TESTING_DATA, testing_data, delimiter=",")
# numpy.savetxt(HOG_TESTING_LABELS, testing_labels, fmt='%i', delimiter=",")