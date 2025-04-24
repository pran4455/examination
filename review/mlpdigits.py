import numpy as np
import idx2numpy
from sklearn.neural_network import MLPClassifier

train_data = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
test_data = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')


model = MLPClassifier(solver='adam',activation='relu')

train_data = train_data.reshape(train_data.shape[0],-1)