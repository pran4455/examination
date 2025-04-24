from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("spam.csv", encoding='latin1')
data = data.iloc[:,:2]
x = data["v2"]
y = data["v1"]

y = y.map({"ham":0,"spam":1})

vectorize = TfidfVectorizer()

x = vectorize.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

model = MultinomialNB()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

print(accuracy_score(ypred,ytest))