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

HOG_TRAINING_DATA = 'data/hog_training_data.npy'
HOG_TRAINING_LABELS = 'data/hog_training_labels.npy'
HOG_TESTING_DATA = 'data/hog_testing_data.npy'
HOG_TESTING_LABELS = 'data/hog_testing_labels.npy'

def print_usage():
    print("Usage:")
    print("  python benchmark [run]")

def run_ann():

    train_labels = numpy.load(HOG_TRAINING_LABELS)
    train_features = numpy.load(HOG_TRAINING_DATA)
    test_labels = numpy.load(HOG_TESTING_LABELS)
    test_features = numpy.load(HOG_TESTING_DATA)

    total_features = numpy.concatenate((train_features, test_features), axis=0)
    total_labels = numpy.concatenate((train_labels, test_labels), axis=0)

    X = numpy.array(total_features)
    Y = numpy.array(total_labels)
    Y = Y - 1
    D = X.shape[1]
    F = len(numpy.unique(Y))
    N = len(X)

    # Preprocess data (normalization and 1-of-c encoding)
    stds = X.std(axis=0)
    for i in range (0, len(stds)):
      if stds[i] == 0:
        stds[i] = 1
    X = (X - X.mean(axis=0)) / stds
    T = numpy.zeros((N, F))
    T[(range(N), Y)] = 1.0

    # Setup network
    net = Net()
    net.set_regularization(0.01, 0.01, 0)
    net.input_layer(D)
    net.fully_connected_layer(100, Activation.LOGISTIC)
    net.output_layer(F, Activation.SOFTMAX)
    net.set_error_function(Error.CE)

    # Split dataset into training set and validation set and make sure that
    # each class is equally distributed in the datasets
    X1 = numpy.vstack((X[0:(N/2)]))
    T1 = numpy.vstack((T[0:(N/2)]))
    training_set = DataSet(X1, T1)
    X2 = numpy.vstack((X[(N/2):]))
    T2 = numpy.vstack((T[(N/2):]))
    validation_set = DataSet(X2, T2)

    # Train for 30 episodes (with tuned parameters for MBSGD)
    optimizer = MBSGD({"maximal_iterations": 30}, learning_rate=0.9,
        learning_rate_decay=0.999, min_learning_rate=0.001, momentum=0.5,
        batch_size=128)
    Log.set_info() # Deactivate debug output
    optimizer.optimize(net, training_set)

    print("TF data set has %d inputs, %d classes and %d examples" % (D, F, N))
    print("The data has been split up input training and validation set.")
    training_percent = float(classification_hits(net, training_set)) / len(X1)
    testing_percent = float(classification_hits(net, validation_set)) / len(X2)
    print("Correct predictions on training set: %d/%d, and percent is: %f"
          % (classification_hits(net, training_set), len(X1), training_percent))
    print("Confusion matrix:")
    print(confusion_matrix(net, training_set)[0])
    print("Correct predictions on test set: %d/%d, and percent is: %f"
          % (classification_hits(net, validation_set), len(X2), testing_percent))
    print("Confusion matrix:")
    print(confusion_matrix(net, validation_set)[0])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "run":
          run_ann()

        else:
            print_usage()
            exit(1)