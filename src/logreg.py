from sklearn import preprocessing, linear_model
import numpy

HOG_TRAINING_DATA = 'data/hog_training_data.npy'
HOG_TRAINING_LABELS = 'data/hog_training_labels.npy'
HOG_TESTING_DATA = 'data/hog_testing_data.npy'
HOG_TESTING_LABELS = 'data/hog_testing_labels.npy'

IMG_TRAINING_DATA = 'data/img_training_data.npy'
IMG_TRAINING_LABELS = 'data/img_training_labels.npy'
IMG_TESTING_DATA = 'data/img_testing_data.npy'
IMG_TESTING_LABELS = 'data/img_testing_labels.npy'

TRAINING_DATA = ''
TRAINING_LABELS = ''
TESTING_DATA = ''
TESTING_LABELS = ''

IMG2D_TRAINING_DATA = 'data/img2d_training_data.npy'
IMG2D_TRAINING_LABELS = 'data/img2d_training_labels.npy'
IMG2D_TESTING_DATA = 'data/img2d_testing_data.npy'
IMG2D_TESTING_LABELS = 'data/img2d_testing_labels.npy'

#Set this flag if we want to train on HOG features
HOG = False

def get_training_set():
  
  train_labels = open(HOG_TRAINING_LABELS)
  train_features = open(HOG_TRAINING_DATA)

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

  test_labels = open(HOG_TESTING_LABELS)
  test_features = open(HOG_TESTING_DATA)

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
  labels = []
  features = []
  if HOG:
    labels = numpy.load(HOG_TRAINING_LABELS)
    features = numpy.load(HOG_TRAINING_DATA)
  else:
    labels = numpy.load(IMG_TRAINING_LABELS)
    features = numpy.load(IMG_TRAINING_DATA)    
  features.tolist()

  print labels
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

  model, scaler = train()

  labels = []
  features = []

  if HOG:
    labels = numpy.load(HOG_TESTING_LABELS)
    features = numpy.load(HOG_TESTING_DATA)
  else:
    labels = numpy.load(IMG_TESTING_LABELS)
    features = numpy.load(IMG_TESTING_DATA)    

  num_correct = 0
  num_wrong = 0
  for i in range(0, len(features)):
    feature_vector = features[i]
    data = []
    data.append(feature_vector)
    data = scaler.transform(data)
    data = preprocessing.normalize(data)
    prediction = model.predict(data)
    print "Predicted " + convert(int(prediction[0])) + " and actually " + convert(int(labels[i]))
    if prediction[0] == labels[i] or inverse(prediction[0]) == labels[i]:
      num_correct = num_correct + 1
    else:
      num_wrong = num_wrong + 1
  print (float(num_correct) / (num_correct + num_wrong))

def cnn_training_set():
  
  train_labels = open(IMG2D_TRAINING_LABELS)
  train_features = open(IMG2D_TRAINING_DATA)

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

def cnn_testing_set():

  test_labels = open(IMG2D_TESTING_LABELS)
  test_features = open(IMG2D_TESTING_DATA)

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

def convert(i):
  if i >= 1 and i <= 10:
    return str(i - 1)
  if i >= 11 and i <= 36:
    return chr(i + 54)
  if i >= 37:
    return chr(i + 60) 

def inverse(i):
  if i >= 1 and i <= 10:
    return i
  if i >= 11 and i <= 36:
    return i + 26
  if i >= 37:
    return i - 26

# test()
