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
USE_BAD = False

training_data = []
training_labels = []
validation_data = []
validation_labels = []
testing_data = []
testing_labels = []

# Range endpoint should be 63, but leaving it as 2 for testing purposes.
for i in range (1, 2):
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
		endpoints.append(len(files)/3)
		endpoints.append(2*len(files)/3)

		for j in range(len(files)):
			filename = directory + files[i]
			img = Image.open(open(filename))

			# convert to B&W, not sure if this works yet
			img = img.convert('L')
			img = numpy.asarray(img, dtype='float64') / 256

			if j >= 0 and j < endpoints[0]:
				training_data.append(img)
				training_labels.append(i)
			elif j >= endpoints[0] and j < endpoints[1]:
				validation_data.append(img)
				validation_labels.append(i)
			elif j >= endpoints[1]:
				testing_data.append(img)
				testing_labels.append(i)

training_data = numpy.asarray(training_data)
training_labels = numpy.asarray(training_labels)
validation_data = numpy.asarray(validation_data)
validation_labels = numpy.asarray(validation_labels)
testing_data = numpy.asarray(testing_data)
testing_labels = numpy.asarray(testing_labels)

final_data_set = ((training_data, training_labels), (validation_data, validation_labels), (testing_data, testing_labels))
print final_data_set[0][0][0][len(final_data_set[0][0][0]) - 1]
#cPickle.dump(final_data_set, 'single_char_data.pkl')

# don't forget to zip!

