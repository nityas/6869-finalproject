"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

import os
import sys
import time
import gzip

import numpy
import itertools

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


import lenet
from lenet.LogisticRegression import *
from lenet.LeNetConvPoolLayer import *
from lenet.HiddenLayer import *
from logreg import cnn_training_set
from logreg import cnn_testing_set


HOG_TRAINING_DATA = 'data/hog_training_data.npy'
HOG_TRAINING_LABELS = 'data/hog_training_labels.npy'
HOG_TESTING_DATA = 'data/hog_testing_data.npy'
HOG_TESTING_LABELS = 'data/hog_testing_labels.npy'

IMG2D_TRAINING_DATA = 'data/img2d_training_data.npy'
IMG2D_TRAINING_LABELS = 'data/img2d_training_labels.npy'
IMG2D_TESTING_DATA = 'data/img2d_testing_data.npy'
IMG2D_TESTING_LABELS = 'data/img2d_testing_labels.npy'

def evaluate_lenet(learning_rate=0.1, n_epochs=200,
                    dataset='res/mnist.pkl',
                    nkerns=[20, 50], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    train_set_y_np = numpy.load(IMG2D_TRAINING_LABELS)
    
    train_set_x_np = numpy.load(IMG2D_TRAINING_DATA)
    train_set_x = theano.shared(value=train_set_x_np, name='train_set_x')

    test_set_y_np = numpy.load(IMG2D_TESTING_LABELS)
    test_set_y = theano.shared(value=test_set_y_np, name='test_set_y')
    
    test_set_x_np = numpy.load(IMG2D_TESTING_DATA)
    test_set_x = theano.shared(value=test_set_x_np, name='test_set_x')

    # F = len(numpy.unique(train_set_y_np))
    # N = len(train_set_x_np)


    
    # tr_y = numpy.zeros((N, F))
    # tr_y[(range(N), train_set_y_np-1)] = 1.0
    train_set_y = theano.shared(value=train_set_y_np, name='train_set_y', broadcastable=(False, True))
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_set_x_np)
    n_test_batches = len(test_set_x_np)
    n_train_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar('index')  # index to a [mini]batch
    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )   
    get_errors = theano.function(
        [index],
        layer3.errors(y,frac=True),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    # validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    maxiter = 1000

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print "epoch ", epoch
        old_impr = test_score
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            test_score = [get_errors(i) for i in xrange(n_test_batches)]

            pred = test_score[0][0]
            actual = test_score[0][1]

            n = len(numpy.unique(actual))
            test_score = float(numpy.mean(test_losses))
            print 'iter ', iter,': accuracy= ', test_score[0]
            print "Confusion Matrix:"
            print numpy.array([zip(actual,pred).count(x) for x in itertools.product(list(set(actual)),repeat=2)]).reshape(n,n)
        
        if test_score-old_impr < 0.01:
            done_looping = True
            break

    end_time = time.clock()
    print('Optimization complete.')
    print('with test performance %f %%' % (test_score * 100.))

if __name__ == '__main__':
    evaluate_lenet(dataset='English')
