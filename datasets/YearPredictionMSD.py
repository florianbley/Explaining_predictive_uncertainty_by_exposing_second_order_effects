import sys

import sklearn.datasets

sys.path.insert(0, 'data')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def create_data(dataset_path):
    df = pd.read_csv('YearPredictionMSD/YearPredictionMSD.txt', header=None)

    # Split the dataframe into X and y dataframes
    df_X = df.iloc[:, 1:]
    df_y = df.iloc[:, :1]

    # Enumerate the input variables
    input_variables = [(i + 1, col) for i, col in enumerate(df_X.columns)]
    # Name the columns of the df_X dataframe
    df_X.columns = ['var{}'.format(i) for i, _ in input_variables]

    test_cutoff = 463715
    df_X_train, df_y_train = df_X.iloc[:test_cutoff], df_y.iloc[:test_cutoff]
    df_X_test, df_y_test = df_X.iloc[test_cutoff:], df_y.iloc[test_cutoff:]

    input_scaler = sklearn.preprocessing.StandardScaler().fit(df_X_train)
    output_scaler = sklearn.preprocessing.StandardScaler().fit(df_y_train)

    df_X_train[df_X.columns], df_X_test[df_X.columns] = input_scaler.transform(df_X_train[df_X.columns]), input_scaler.transform(df_X_test[df_X.columns])
    df_y_train[df_y.columns], df_y_test[df_y.columns] = output_scaler.transform(df_y_train[df_y.columns]), output_scaler.transform(df_y_test[df_y.columns])

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    df_X_train.to_pickle(dataset_path + "/df_train_input")
    df_y_train.to_pickle(dataset_path + "/df_train_output")
    df_X_test.to_pickle(dataset_path + "/df_test_input")
    df_y_test.to_pickle(dataset_path + "/df_test_output")
    return df_X_train.values, df_X_test.values, df_y_train.values, df_y_test.values


def load_data(dataset_path):
    X_train = pd.read_pickle(dataset_path + "/df_train_input").values
    y_train = pd.read_pickle(dataset_path + "/df_train_output").values
    X_test = pd.read_pickle(dataset_path + "/df_test_input").values
    y_test = pd.read_pickle(dataset_path + "/df_test_output").values
    return X_train, X_test, y_train, y_test


def check_if_dataset_exists(dataset_path):
    data_root = os.getcwd()
    for data_object in ["df_train_input", "df_test_input", "df_train_output", "df_test_output"]:
        object_exists = os.path.exists(os.path.join(data_root, dataset_path+'/'+data_object))
        if not object_exists:
            return False
    return True


def serve_dataset():
    dataset_path = get_dataset_path()
    dataset_exists = check_if_dataset_exists(dataset_path)
    if dataset_exists:
        X_train, X_test, y_train, y_test = load_data(dataset_path)
    else:
        X_train, X_test, y_train, y_test = create_data(dataset_path)

    n_train = int(X_train.shape[0] * 0.9)
    X_train, X_val = X_train[:n_train], X_train[n_train:]
    y_train, y_val = y_train[:n_train], y_train[n_train:]
    return X_train, X_test, X_val, y_train, y_test, y_val


def get_dataset_path():
    data_path = "YearPredictionMSD"
    data_root = os.getcwd()
    if "datasets" not in data_root:
        data_path = "datasets/"+data_path
    dataset_path = data_path
    return dataset_path


if __name__ == "__main__":
    dataset_path = get_dataset_path()
    create_data(dataset_path)
    X_train, X_test, y_train, y_test = serve_dataset()
    print("Done")