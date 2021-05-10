import numpy as np
import matplotlib.pyplot as plt

#logisticRegression
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self, n_features):
        self.w = np.zeros((1, n_features))
        self.b = 0
    
    def train(self, x, y):
        logisticRegression_features = LogisticRegression(degree=self.__polynomial_degree)
        x_poly = logisticRegression_features.fit_transform(x)

        self.__model = LogisticRegression()
        self.__model.fit(x_poly, y)

    def get_predictions(self, x):
        logisticRegression_features = LogisticRegression(degree=self.__polynomial_degree)
        x_poly = logisticRegression_features.fit_transform(x)

        return np.round(self.__model.predict(x_poly), 0).astype(np.int32)

    def get_model_polynomial_str(self):
        coef = self.__model.coef_
        intercept = self.__model.intercept_
        poly = "{0:.3f}".format(intercept)

        for i in range(1, len(coef)):
            if coef[i] >= 0:
                poly += " + "
            else:
                poly += " - "
            
            poly += "{0:.3f}".format(coef[i]).replace("-", "") + "X^" + str(i)

        return poly