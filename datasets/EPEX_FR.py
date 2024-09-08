import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


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
        raise ValueError("Format {} not supported".format(format))


def get_future_data(df, lookback):
    prediction_dates = []
    future_prices = []
    load_forecasts = []
    gen_forecasts = []
    i = 0
    for prediction_date, row in df.iloc[:-lookback].iterrows():

        ts = pd.Timestamp(prediction_date)

        one_hour = pd.DateOffset(hours=1)
        one_day = pd.DateOffset(hours=24)

        future_entries = df.loc[str(ts+one_hour):str(ts+one_day)]
        #future_prices.append(future_entries[""])
        prediction_dates.append(ts)
        future_prices.append(future_entries["Prices"].values)
        load_forecasts.append(future_entries["System load forecast"].values)
        gen_forecasts.append(future_entries["Generation forecast"].values)
        i = i + 1

    output_cols = ["DA_price_tplus_{}".format(i+1) for i in range(24)]
    df_out = pd.DataFrame(columns=output_cols, data=future_prices)
    df_out["prediction_date"] = prediction_dates
    df_out.set_index("prediction_date", inplace=True)

    gen_forecast_cols = ["DA_gen_tplus_{}".format(i+1) for i in range(24)]
    df_gen = pd.DataFrame(columns=gen_forecast_cols, data=gen_forecasts)
    df_gen["prediction_date"] = prediction_dates
    df_gen.set_index("prediction_date", inplace=True)

    load_forecast_cols = ["DA_load_tplus_{}".format(i+1) for i in range(24)]
    df_load = pd.DataFrame(columns=load_forecast_cols, data=load_forecasts)
    df_load["prediction_date"] = prediction_dates
    df_load.set_index("prediction_date", inplace=True)

    df_forecasts = df_load.join(df_gen)
    return df_out, df_forecasts, df_gen, df_load


def get_past_data(df, lookback_hours):
    prediction_dates = []
    previous_prices = []
    previous_load_forecasts = []
    previous_gen_forecasts = []
    for prediction_date, row in df.iloc[lookback_hours:].iterrows():
        ts = pd.Timestamp(prediction_date)
        lookback = pd.DateOffset(hours=lookback_hours-1)
        ts_lookback = ts - lookback
        lookback_entries = df.loc[str(ts_lookback):str(ts)]

        previous_prices.append(lookback_entries["Prices"].values)
        previous_load_forecasts.append(lookback_entries["System load forecast"].values)
        previous_gen_forecasts.append(lookback_entries["Generation forecast"].values)
        prediction_dates.append(ts)

    past_prices_cols = ["DA_price_tminus_{}".format(i) for i in range(lookback_hours)][::-1]
    df_past_prices = pd.DataFrame(columns=past_prices_cols, data=previous_prices)
    df_past_prices["prediction_date"] = prediction_dates
    df_past_prices.set_index("prediction_date", inplace=True)

    past_gen_forecast_cols = ["DA_gen_tminus_{}".format(i) for i in range(lookback_hours)][::-1]
    df_gen = pd.DataFrame(columns=past_gen_forecast_cols, data=previous_gen_forecasts)
    df_gen["prediction_date"] = prediction_dates
    df_gen.set_index("prediction_date", inplace=True)

    past_load_forecast_cols = ["DA_load_tminus_{}".format(i) for i in range(lookback_hours)][::-1]
    df_load = pd.DataFrame(columns=past_load_forecast_cols, data=previous_load_forecasts)
    df_load["prediction_date"] = prediction_dates
    df_load.set_index("prediction_date", inplace=True)

    return df_past_prices, df_gen, df_load


def get_dataset_path(lookback):
    data_path = "EPEX-FR"
    data_root = os.getcwd()
    if "datasets" not in data_root:
        data_path = "datasets/" + data_path
    dataset_path = data_path + "/lookback_{}".format(lookback)
    return dataset_path


def make_tabular_data(df_past_prices, df_forecasts, df_target, test_cutoff_date, lookback):
    df_tabluar_data = df_past_prices.join(df_forecasts, how="inner").join(df_target, how="inner")

    df_train_data_tabular = df_tabluar_data.loc[:str(test_cutoff_date)]
    df_test_data_tabular = df_tabluar_data.loc[str(test_cutoff_date):].iloc[1:]

    output_cols = ["DA_price_tplus_{}".format(i + 1) for i in range(24)]
    df_train_input_tabular = df_train_data_tabular.drop(columns=output_cols)
    df_train_output_tabular = df_train_data_tabular[output_cols]

    df_test_input_tabular = df_test_data_tabular.drop(columns=output_cols)
    df_test_output_tabular = df_test_data_tabular[output_cols]

    path = get_dataset_path(lookback)
    if not os.path.exists(path):
        os.makedirs(path)

    df_train_input_tabular.to_pickle(path + "/df_train_input_tabular")
    df_train_output_tabular.to_pickle(path + "/df_train_output_tabular")
    df_test_input_tabular.to_pickle(path + "/df_test_input_tabular")
    df_test_output_tabular.to_pickle(path + "/df_test_output_tabular")

    return df_train_input_tabular, df_train_output_tabular, df_test_input_tabular, df_test_output_tabular


def make_channel_data(df_past_prices, df_past_gen, df_past_load, df_target, df_gen_forecasts, df_load_forecasts,
                      lookback, test_cutoff_date):

    # channel split data
    # first join past 24 hour load and load forecasts, past gen and gen forecasts and past prices
    past_load_columns = ["DA_load_tminus_{}".format(i) for i in range(lookback - 24)][::-1]
    past_gen_columns = ["DA_gen_tminus_{}".format(i) for i in range(lookback - 24)][::-1]

    df_gen = df_past_gen[past_gen_columns].join(df_gen_forecasts, how="inner")
    df_load = df_past_load[past_load_columns].join(df_load_forecasts, how="inner")

    joint_index = df_target.index.join(df_gen.index, how="inner").join(df_load.index, how="inner").join(
        df_past_prices.index, how="inner")
    df_target = df_target.loc[joint_index]
    df_gen = df_gen.loc[joint_index]
    df_load = df_load.loc[joint_index]
    df_past_prices = df_past_prices.loc[joint_index]

    # check that indices of df_target, df_gen, df_load, df_past_prices are the same
    assert df_target.index.equals(df_gen.index)
    assert df_target.index.equals(df_load.index)
    assert df_target.index.equals(df_past_prices.index)

    train_inds = df_target.index <= test_cutoff_date
    test_inds = df_target.index > test_cutoff_date

    # X is of shape (n_samples, n_channels=3, n_features=lookback)
    X_train = np.stack(
        [df_load.loc[train_inds].values, df_gen.loc[train_inds].values, df_past_prices.loc[train_inds].values],
        axis=1)
    X_test = np.stack(
        [df_load.loc[test_inds].values, df_gen.loc[test_inds].values, df_past_prices.loc[test_inds].values], axis=1)

    y_train = df_target.loc[train_inds].values
    y_test = df_target.loc[test_inds].values

    # save all as pickle files
    path = get_dataset_path(lookback)
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(path + "/X_train_channel", X_train)
    np.save(path + "/X_test_channel", X_test)
    np.save(path + "/y_train_channel", y_train)
    np.save(path + "/y_test_channel", y_test)
    return X_train, X_test, y_train, y_test


def create_data(lookback, format="tabular"):
    # original dataset FR.csv was produced with the epftoolbox repository that can be found at
    # https://github.com/jeslago/epftoolbox
    df = pd.read_csv("datasets/EPEX-FR/FR.csv", skipinitialspace=True)
    df.set_index("Date", inplace=True)

    last_date = pd.Timestamp(df.index[-1])
    one_year = pd.DateOffset(years=1)
    test_cutoff_date = last_date - one_year

    df_train = df.loc[:str(test_cutoff_date)]
    scaler = StandardScaler()
    scaler.fit(df_train)
    df[df.columns] = scaler.transform(df[df.columns])

    df_past_prices, df_past_gen, df_past_load = get_past_data(df, lookback_hours=lookback)
    df_target, df_forecasts, df_gen_forecasts, df_load_forecasts = get_future_data(df, lookback)

    X_train_tab, X_test_tab, y_train_tab, y_test_tab = make_tabular_data(df_past_prices, df_forecasts, df_target, test_cutoff_date, lookback)

    X_train_channel, X_test_channel, y_train_channel, y_test_channel = make_channel_data(
        df_past_prices, df_past_gen, df_past_load, df_target,
        df_gen_forecasts, df_load_forecasts, lookback, test_cutoff_date)

    if format == "tabular":
        return X_train_tab, X_test_tab, y_train_tab, y_test_tab
    elif format == "channel":
        return X_train_channel, X_test_channel, y_train_channel, y_test_channel


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


def serve_dataset(lookback, format="tabular"):
    dataset_path = get_dataset_path(lookback)
    dataset_exists = check_if_dataset_exists(dataset_path, format)
    if dataset_exists:
        X_train, X_test, y_train, y_test = load_data(dataset_path, format)

    else:
        X_train, X_test, y_train, y_test = create_data(lookback, format)

    # reserve last 10% of the training data for validation
    n_train = int(X_train.shape[0] * 0.9)
    X_train, X_val = X_train[:n_train], X_train[n_train:]
    y_train, y_val = y_train[:n_train], y_train[n_train:]

    return X_train, X_test, X_val, y_train, y_test, y_val

if __name__ == "__main__":
    format = "channel"
    # move path one up
    os.chdir("..")

    # print current working directory
    print(os.getcwd())

    serve_dataset(48, format)
