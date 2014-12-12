import sys
try:
    from sklearn import datasets
except:
    print("scikit-learn is required to run this example.")
    exit(1)
try:
    from openann import *
except:
    print("OpenANN Python bindings are not installed!")
    exit(1)

#NOTE: LABELS ARE 0-INDEXED, UNLIKE WITH LOGISTIC REGRESSION

IMG2D_TRAINING_DATA = 'data2/img_training_data.npy'
IMG2D_TRAINING_LABELS = 'data2/img_training_labels.npy'
IMG2D_TESTING_DATA = 'data2/img_testing_data.npy'
IMG2D_TESTING_LABELS = 'data2/img_testing_labels.npy'

HOG = False

def print_usage():
    print("Usage:")
    print("  python benchmark [run]")

def run_cnn():

    train_labels = []
    train_features = []
    test_labels = []
    test_features = []

    train_labels = numpy.load(IMG2D_TRAINING_LABELS)
    train_features = numpy.load(IMG2D_TRAINING_DATA)
    test_labels = numpy.load(IMG2D_TESTING_LABELS)
    test_features = numpy.load(IMG2D_TESTING_DATA)        

    total_features = numpy.concatenate((train_features, test_features), axis=0)
    total_labels = numpy.concatenate((train_labels, test_labels), axis=0)

    X = numpy.array(total_features)
    Y = numpy.array(total_labels)
    Y = Y - 1
    print "X shape is "
    print X.shape
    D = X.shape[1]
    F = len(numpy.unique(Y))
    N = len(X)

    # Preprocess data (normalization and 1-of-c encoding)
    #Note: took this out. See ann.py 

    stds = X.std(axis=0)
    # for i in range (0, len(stds)):
    #   if stds[i] == 0:
    #     stds[i] = 1
    X = (X - X.mean(axis=0)) / stds
    T = numpy.zeros((N, F))
    T[(range(N), Y)] = 1.0


    # Setup network
    #currently at 53%
    net = Net()
    net.set_regularization(0, 7, 0)
    net.input_layer(1, 75, 75)
    net.dropout_layer(0.2)
    net.convolutional_layer(66, 10, 10, Activation.RECTIFIER, 0.05)
    net.maxpooling_layer(2, 2)
    net.dropout_layer(0.2)
    net.maxpooling_layer(2, 2)
    net.convolutional_layer(66, 10, 10, Activation.RECTIFIER, 0.05)
    print "done setting up network"
    # net.dropout_layer(0.2)
    # net.maxpooling_layer(2, 2)
    # net.fully_connected_layer(200, Activation.RECTIFIER, 0.05)
    # net.dropout_layer(0.4)
    # net.fully_connected_layer(150, Activation.RECTIFIER, 0.05)
    #net.dropout_layer(0.4);
    net.output_layer(F, Activation.SOFTMAX)
    net.set_error_function(Error.CE)

    X1 = numpy.vstack((X[0:(N/2)]))
    print X1.shape
    T1 = numpy.vstack((T[0:(N/2)]))
    print T1.shape
    training_set = DataSet(X1, T1)
    X2 = numpy.vstack((X[(N/2):]))
    T2 = numpy.vstack((T[(N/2):]))
    validation_set = DataSet(X2, T2)

    # Train for 30 episodes (with tuned parameters for MBSGD)
    print "creating optimizer"
    optimizer = MBSGD({"maximal_iterations": 10}, learning_rate=0.05,
        learning_rate_decay=0.999, min_learning_rate=0.001, momentum=0.5,
        batch_size=128)
    #Log.set_info() # Deactivate debug output

    print "beginning training"
    optimizer.optimize(net, training_set)

    print "done training"
    num_right_training = classification_hits(net, training_set)
    num_right_testing = classification_hits(net, validation_set)

    print("TF data set has %d inputs, %d classes and %d examples" % (D, F, N))
    print("The data has been split up input training and validation set.")
    training_percent = float(num_right_training) / len(X1)
    testing_percent = float(num_right_testing) / len(X2)
    print("Correct predictions on training set: %d/%d, and percent is: %f"
          % (num_right_training, len(X1), training_percent))
    #print("Confusion matrix:")
    #print(confusion_matrix(net, training_set))
    print("Correct predictions on test set: %d/%d, and percent is: %f"
          % (num_right_testing, len(X2), testing_percent))
    #print("Confusion matrix:")
    #print(confusion_matrix(net, validation_set))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "run":
          run_cnn()

        else:
            print_usage()
            exit(1)