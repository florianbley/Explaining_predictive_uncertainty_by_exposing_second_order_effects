import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def check_if_dataset_exists(dataset_path):
    data_root = os.getcwd()
    for data_object in ["df_train_input", "df_test_input", "df_train_output", "df_test_output"]:
        object_exists = os.path.exists(os.path.join(data_root, dataset_path+'/'+data_object))
        if not object_exists:
            return False
    return True

def get_rawdata_paths():
    data_path = "Wine Quality"
    raw_data_path_red = data_path + "/winequality-red_komma.csv"
    raw_data_path_white = data_path + "/winequality-white_komma.csv"
    root_dir = os.getcwd()
    if "datasets" not in root_dir:
        raw_data_path_red = "datasets/" + raw_data_path_red
        raw_data_path_white = "datasets/" + raw_data_path_white
    return raw_data_path_red, raw_data_path_white

def get_dataset_path():
    data_path = "Wine Quality"
    data_root = os.getcwd()
    if "datasets" not in data_root:
        data_path = "datasets/" + data_path
    return data_path

def create_data(dataset_path, dataset_path_red, dataset_path_white):
    np.random.seed(0)
    df_red = pd.read_csv(dataset_path_red, delimiter=";", decimal=",")
    df_white = pd.read_csv(dataset_path_white, delimiter=";", decimal=",")
    df = pd.concat([df_red, df_white])

    # random shuffling
    shuffle_int = np.random.permutation(range(len(df)))
    df = df.iloc[shuffle_int]

    test_set_ratio = 0.2
    len_test_set = int(test_set_ratio * len(df))

    df_train = df.iloc[:-len_test_set]
    scaler = StandardScaler()
    scaler.fit(df_train)
    df[df.columns] = scaler.transform(df[df.columns])

    input_cols = [col for col in df.columns if col != "quality"]
    df_input = df[input_cols]
    df_target = df["quality"]

    df_X_train, df_X_test = df_input[:-len_test_set], df_input[-len_test_set:]
    df_y_train, df_y_test = df_target[:-len_test_set], df_target[-len_test_set:]

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


def serve_dataset():
    dataset_path = get_dataset_path()
    raw_data_path_red, raw_data_path_white = get_rawdata_paths()
    dataset_exists = check_if_dataset_exists(dataset_path)
    if dataset_exists:
        X_train, X_test, y_train, y_test = load_data(dataset_path)

    else:
        X_train, X_test, y_train, y_test = create_data(dataset_path, raw_data_path_red, raw_data_path_white)

    n_train = int(X_train.shape[0] * 0.9)
    X_train, X_val = X_train[:n_train], X_train[n_train:]
    y_train, y_val = y_train[:n_train], y_train[n_train:]
    return X_train, X_test, X_val, y_train[:, None], y_test[:, None], y_val[:, None]

if __name__ == "__main__":
    np.random.seed(0)
    X_train, X_test, y_train, y_test = serve_dataset()
    print("Done")