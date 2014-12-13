import gzip
import cPickle
from PIL import Image

#a, b, c = cPickle.load(gzip.open('mnist.pkl', 'rb'))
#d, e, f = cPickle.load(gzip.open('EnglishHandwritten.gz', 'rb'))
#x, y, z = cPickle.load(gzip.open('EnglishNatural.gz', 'rb'))
xx, yy, zz = cPickle.load(gzip.open('SegTest.gz', 'rb'))

def triple(x):
    return (x, x, x)

def show_img(img, width, height):
    i = Image.new('RGB', (width, height), 'black')
    p = i.load()
    for w in range(width):
        for h in range(height):
            p[w, h] = triple(int(img[w * height + h]*255))
    return i
