from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os

traindir = r'train/'
testdir = r'test/'

traindata = []
testdata = []
for i in ['pos','neg']:

    path = os.path.join(traindir,i)
    for j in range(12500):
        fname = f'{i}'