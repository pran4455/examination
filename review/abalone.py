import numpy as np
import pandas as pd

class LinearRegression:

    def __init__(self,n_iter,learning,X,y):
        
        self.n_iter = n_iter
        self.lrate = learning
        self.X = X
        self.y = y
        self.m = 0
        self.c = 0
        self.n = len(y)

    def fit(self):
        
        for i in range(self.n_iter):
            ypred = self.m * self.X + self.c
            dm = -(2 / self.n)*np.sum(self.X * (ypred - self.y))
            dc = -(2 / self.n)*np.sum(ypred - self.y)

            self.m -= self.lrate*dm
            self.c -= self.lrate*dc
    
    def pred(self,x):

        return self.m * x + self.c
    
if __name__ == '__main__':

    # data = pd.read_csv("abalone.data")
    # n = len(data)

    # print(data.columns)

    # y = data["age"]
    # x = data.drop(labels=["age"])

    # print(data.columns)

    # split = int(n * 0.8)
    # x_train = x[:split]
    # x_test = x[split:]

    # y_train = x[:split]
    # y_test = y[split:]

    # model = LinearRegression(1000,1e-5,x_train,y_train)
    # model.fit()
    # print(model.pred(x_test[0]))
    # print(y_test[0])
    columns = ["Sex","Lneght","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]

    data = pd.read_csv("abalone.data",header=None,names=columns)
    data["Sex"] = data["Sex"].map({"M":0, "I":1, "F":2})