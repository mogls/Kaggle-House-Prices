import numpy as np
import pandas as pd

dataset = pd.read_csv("train.csv")
x_train = dataset.iloc[0:, 1:-1].values    
y_train = dataset.iloc[0:, -1].values

dataset = pd.read_csv("test.csv")
x_test = dataset.iloc[0:, 1:-1].values    
y_test = dataset.iloc[0:, -1].values


# take care of missing data

from sklearn.impute import SimpleImputer

data_replacer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_replacer.fit(x_train[:, :])
x_train[:, :] = data_replacer.transform(x_train[:, :])
x_test[:, :] = data_replacer.fit_transform(x_test[:, :])


# encoding categorical

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 34, 38, 39, 40, 41, 52, 54, 56, 57, 59, 62, 63, 64, 71, 72, 73, 77, 78])], remainder='passthrough')
x_train = np.array(ct.fit_transform(x_train))
x_test = np.array(ct.fit_transform(x_test))
