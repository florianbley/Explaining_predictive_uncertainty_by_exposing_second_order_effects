import sys
sys.path.insert(0, 'data')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def tabular_to_channelized(df, orig_cols, lags):
    # now create a data version which contains the lagged input data in a channel format
    # iterate over the original column names and list all lagged column names in separate lists
    list_of_channels = []
    for col in orig_cols:
        lagged_cols = [f"{col} (t-{lag})" for lag in lags[1:]]

        channel = df[lagged_cols].values
        list_of_channels.append(channel)
    X_channel = np.stack(list_of_channels, axis=1)
    return X_channel

def create_data(lookback, format):
    path = "datasets/Seoul Bike/SeoulBikeData.csv"
    df = pd.read_csv(path, delimiter=",", decimal=".", encoding = "ISO-8859-1")
    df["Datetime"] = pd.to_datetime(df.Date, dayfirst=True) + df.Hour.astype("timedelta64[h]")
    df = df.drop(columns=["Functioning Day", "Holiday", "Seasons", "Date", "Hour"], axis=0)
    df = df.set_index("Datetime")

    testset_date_cutoff = df.index[int(.75*len(df))]
    orig_cols = df.columns
    df_train = df.loc[:testset_date_cutoff]
    scaler = StandardScaler()
    scaler.fit(df_train)
    df[orig_cols] = scaler.transform(df[orig_cols])

    lags = range(0, lookback+1)
    df_lagged = pd.DataFrame()
    df_lagged = df_lagged.assign(**{
        f'{col} (t-{lag})': df[col].shift(lag)
        for lag in lags
        for col in df
    })
    df_lagged.dropna(axis=0, inplace=True)

    target_col = ["Rented Bike Count (t-0)"]
    input_cols = [col for col in df_lagged.columns if col not in target_col and "(t--1)" not in col]

    df_y = df_lagged[target_col]
    df_X = df_lagged[input_cols]

    df_y_train, df_y_test = df_y[:testset_date_cutoff], df_y[testset_date_cutoff:]
    df_X_train, df_X_test = df_X[:testset_date_cutoff], df_X[testset_date_cutoff:]

    dataset_path = get_dataset_path(lookback)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    df_X_train.to_pickle(dataset_path + "/df_train_input_tabular")
    df_y_train.to_pickle(dataset_path + "/df_train_output_tabular")
    df_X_test.to_pickle(dataset_path + "/df_test_input_tabular")
    df_y_test.to_pickle(dataset_path + "/df_test_output_tabular")

    X_train_channel = tabular_to_channelized(df_X_train, orig_cols, lags)
    X_test_channel = tabular_to_channelized(df_X_test, orig_cols, lags)
    y_train_channel = df_y_train.values
    y_test_channel = df_y_test.values

    np.save(dataset_path + "/X_train_channel", X_train_channel)
    np.save(dataset_path + "/X_test_channel", X_test_channel)
    np.save(dataset_path + "/y_train_channel", y_train_channel)
    np.save(dataset_path + "/y_test_channel", y_test_channel)

    if format == "tabular":
        return df_X_train.values, df_X_test.values, df_y_train.values, df_y_test.values
    elif format == "channel":
        return X_train_channel, X_test_channel, y_train_channel, y_test_channel
    else:
        raise ValueError("format must be either 'tabular' or 'channel'")


def load_tabular_data(dataset_path):
    X_train = pd.read_pickle(dataset_path + "/df_train_input_tabular").values
    y_train = pd.read_pickle(dataset_path + "/df_train_output_tabular").values
    X_test = pd.read_pickle(dataset_path + "/df_test_input_tabular").values
    y_test = pd.read_pickle(dataset_path + "/df_test_output_tabular").values
    return X_train, X_test, y_train, y_test


def load_channel_data(dataset_path):
    X_train = np.load(dataset_path + "/X_train_channel.npy")
    X_test = np.load(dataset_path + "/X_test_channel.npy")
    y_train = np.load(dataset_path + "/y_train_channel.npy")
    y_test = np.load(dataset_path + "/y_test_channel.npy")
    return X_train, X_test, y_train, y_test


def load_data(dataset_path, format="tabular"):
    if format == "tabular":
        return load_tabular_data(dataset_path)
    elif format == "channel":
        return load_channel_data(dataset_path)
    else:
        raise ValueError("format must be either 'tabular' or 'channel'")


def get_dataset_path(lookback):
    data_path = "Seoul Bike"
    data_root = os.getcwd()
    if "datasets" not in data_root:
        data_path = "datasets/"+data_path
    dataset_path = data_path + "/lookback_{}".format(lookback)
    return dataset_path


def check_if_tabular_dataset_exists(dataset_path):
    data_root = os.getcwd()
    for data_object in ["df_train_input", "df_test_input", "df_train_output", "df_test_output"]:
        object_exists = os.path.exists(
            os.path.join(data_root, dataset_path + '/' + data_object + "_{}".format("tabular")))
        if not object_exists:
            return False
    return True


def check_if_channel_dataset_exists(dataset_path):
    data_root = os.getcwd()
    for data_object in ["X_train_channel.npy", "X_test_channel.npy", "y_train_channel.npy", "y_test_channel.npy"]:
        object_exists = os.path.exists(
            os.path.join(data_root, dataset_path + '/' + data_object))
        if not object_exists:
            return False
    return True


def check_if_dataset_exists(dataset_path, format="tabular"):

    if format == "tabular":
        return check_if_tabular_dataset_exists(dataset_path)
    elif format == "channel":
        return check_if_channel_dataset_exists(dataset_path)
    else:
        raise ValueError("Invalid format. Please choose between 'tabular' and 'channel'.")


def serve_dataset(lookback, format="tabular"):
    dataset_path = get_dataset_path(lookback)
    dataset_exists = check_if_dataset_exists(dataset_path, format)
    if dataset_exists:
        X_train, X_test, y_train, y_test = load_data(dataset_path, format)
    else:
        X_train, X_test, y_train, y_test = create_data(lookback, format)

    n_train = int(X_train.shape[0] * 0.9)
    X_train, X_val = X_train[:n_train], X_train[n_train:]
    y_train, y_val = y_train[:n_train], y_train[n_train:]
    return X_train, X_test, X_val, y_train, y_test, y_val

if __name__ == "__main__":
    lookback = 10
    dataset_path = get_dataset_path(lookback)
    # go one path up
    os.chdir("..")
    #X_train, X_test, y_train, y_test = create_data(lookback, "channel")

    X_train, X_test, X_val, y_train, y_test, y_val = serve_dataset(lookback, format="channel")
    print("Done")