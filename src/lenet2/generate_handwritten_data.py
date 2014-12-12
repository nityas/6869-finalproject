import gzip
import cPickle
from os import listdir
from PIL import Image
import numpy as np

output = 'EnglishHandwritten.gz'

images = []
values = []
for group in listdir('data'):
    if '~' in group:
        continue
    print 'beginning group', group
    for png in listdir('data/%s' % group):
        img = Image.open('data/%s/%s' % (group, png))
        img = img.convert('LA') # convert to grayscale
        img.thumbnail((28, 21), Image.ANTIALIAS)
        data = img.load()
        nex = np.zeros((28, 28))
        for i in range(28):
            for j in range(28):
                nex[i, j] = 1.0 - (data[i, j][0] / 255.0 if j < 21 else 1.0);
        images.append(nex.reshape((784,)))
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
