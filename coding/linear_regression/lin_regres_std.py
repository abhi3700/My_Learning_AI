# ----------------------------------------------------------
# Reference: https://hackerstreak.com/linear-regression-with-python-and-numpy/
# ----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------------------------------Read data---------------------------------------------------------------------
url = 'https://raw.githubusercontent.com/Baakchsu/LinearRegression/master/weight-height.csv'
df = pd.read_csv(url)
print(df)

# ------------------------------------------------class "Linear Regression"-----------------------------------------------------
class linear_regression:
    def fit(self, X, Y):
        X = np.array(X).reshape(-1, 1)
        x_shape = X.shape

    def predict(self, X):
        return product
