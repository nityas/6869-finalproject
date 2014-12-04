#CITATION: All data taken from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

import numpy
import sys
import os
from PIL import Image
import cPickle
import gzip

root = 'English/Img/'
bad = 'BadImag/Bmp/'
good = 'GoodImg/Bmp/'

#turn this flag on if we want to test on a harder data set
USE_BAD = false

training_data = []
training_labels = []
validation_data = []
validation_labels = []
testing_data = []
testing_labels = []

for i in range (1, 63):
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
		endpoints[0] = len(files)/3
		endpoints[1] = 2*len(files)/3

		for j in range(len(files)):
			filename = files[i]
			img = Image.open(open(filename))

			# convert to B&W, not sure if this works yet
			img = img.convert('1')
			img = numpy.asarray(img, dtype='float64') / 256

			if j >= 0 and j < endpoints[0]:
				training_data.append(img)
				training_labels.append(i)
			else if j >= endpoints[0] and j < endpoints[1]:
				validation_data.append(img)
				validation_labels.append(i)
			else if j >= endpoints[1] and j < endpoints[2]:
				testing_data.append(img)
				testing_labels.append(i)

training_data = numpy.asarray(training_data)
training_labels = numpy.asarray(training_labels)
validation_data = numpy.asarray(validation_data)
validation_labels = numpy.asarray(validation_labels)
testing_data = numpy.asarray(testing_data)
testing_labels = numpy.asarray(testing_labels)

final_data_set = ((training_data, training_labels), (validation_data, validation_labels), (testing_data, testing_labels))
cPickle.dump(final_data_set, 'single_char_data.pkl')

# don't forget to zip!

