import numpy as np
import pandas as pd

# take care of missing numerical data

def numeric_missing_data(dataset: pd.DataFrame):

    numeric_columns = dataset.select_dtypes(include="number")
    numeric_column_names = list(numeric_columns.columns)
    column_means = [dataset[column].mean() for column in numeric_column_names]
    numeric_to_replace = {column_name:mean_column for (column_name, mean_column) in zip(numeric_column_names, column_means) if column_name != 'Id'}

    dataset.fillna(numeric_to_replace, inplace=True)

    return dataset


# take care of mising non-numeric data

def non_num_missing_data(dataset: pd.DataFrame, minimum_entries: int = 100):

    non_num_columns = dataset.select_dtypes(exclude="number")
    non_num_column_names = list(non_num_columns.columns)
    column_average = [dataset[column].mode().iloc[0] for column in non_num_column_names]
    non_num_to_replace = {column_name:mean_column for (column_name, mean_column) in zip(non_num_column_names, column_average) if dataset[column_name].isna().sum() < minimum_entries}

    dataset.fillna(non_num_to_replace, inplace=True)

    return dataset


#import the datasets

dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")

# take care of missing data

dataset = numeric_missing_data(dataset)
dataset = non_num_missing_data(dataset, 200)
test_dataset = numeric_missing_data(test_dataset)
test_dataset = non_num_missing_data(test_dataset, 200)

# create input

x_train = dataset.iloc[0:, 1:-1].values    
y_train = dataset.iloc[0:, -1].values

x_test = test_dataset.iloc[0:, 1:].values    

# get columns with Non-numeric values:

non_numeric_columns = dataset.select_dtypes(exclude='number')

non_numeric_column_indices = [dataset.columns.get_loc(col)-1 for col in non_numeric_columns]

# encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), non_numeric_column_indices)], remainder='passthrough', sparse_threshold=0)
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)

# scailing

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
X = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

sc_y = StandardScaler()
y = sc_y.fit_transform(y_train.reshape(-1, 1))


# training the model
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# predicting results

results = sc_y.inverse_transform(regressor.predict(x_test).reshape(-1, 1))

# exporting to csv

indexes = list(range(1461, 1461+len(results)))

to_csv = {"Id":indexes, "results":[result[0] for result in results]}

df_to_csv = pd.DataFrame(to_csv).set_index("Id")

df_to_csv.to_csv("./SVM/SVM_Results.csv")