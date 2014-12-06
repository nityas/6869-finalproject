from sklearn import preprocessing, linear_model
import numpy

HOG_TRAINING_DATA = 'data/hog_training_data'
HOG_TRAINING_LABELS = 'data/hog_training_labels'
HOG_TESTING_DATA = 'data/hog_testing_data'
HOG_TESTING_LABELS = 'data/hog_testing_labels'

# HOG_TRAINING_DATA = 'data/hog_training_data.csv'
# HOG_TRAINING_LABELS = 'data/hog_training_labels.csv'
# HOG_TESTING_DATA = 'data/hog_testing_data.csv'
# HOG_TESTING_LABELS = 'data/hog_testing_labels.csv'

NUM_FEATURES = 1500
NUM_TRAINING = 50000 #100000 for first data set

def get_training_set():
  
  train_labels = open(TRAIN_LABEL_PATH)
  train_features = open(TRAIN_FEATURES_PATH)

  labels = []
  features = []

  for line in train_features.readlines():
    feature_vector = line.strip().split(',')
    feature = map(float, feature_vector)
    features.append(feature)
  for line in train_labels.readlines():
    label = map(int, line.strip().split(','))
    labels.append(label[0])
  return labels, features

def get_testing_set():

  test_labels = open(TEST_LABEL_PATH)
  test_features = open(TEST_FEATURES_PATH)

  labels = []
  features = []

  for line in test_features.readlines():
    feature_vector = line.strip().split(',')
    feature = map(float, feature_vector)
    features.append(feature)
  for line in test_labels.readlines():
    label = map(int, line.strip().split(','))
    labels.append(label[0])
  return labels, features


def train():
  print "beginning training"
  #labels, features = get_training_set()
  labels = numpy.load()
  scaler = preprocessing.StandardScaler().fit(features)
  features = scaler.transform(features)
  features = preprocessing.normalize(features)

  model = linear_model.LogisticRegression()
  
  #print numpy.array(labels).shape
  #print numpy.array(features).shape
  model.fit(features, labels)
  print "ending training"
  return model, scaler

def test():
  print "beginning testing"

  model, scaler = train(data_file)
  num_correct = 0
  num_wrong = 0
  for i in range(0, len(features)):
    feature_vector = features[i]
    data = []
    data.append(feature_vector)
    data = scaler.transform(data)
    data = preprocessing.normalize(data)
    prediction = model.predict(data)
    print str(prediction[0]) + " " + str(labels[i])
    if prediction[0] == labels[i]:
      num_correct = num_correct + 1
    else:
      num_wrong = num_wrong + 1
  print (float(num_correct) / (num_correct + num_wrong))

test()