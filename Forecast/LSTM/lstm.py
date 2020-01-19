import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import to_categorical
from numpy import array
from numpy import split
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

EPSILON = 1e-10
np.set_printoptions(precision=3, suppress=True)


"""
Taken from https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
github user Boris Shishov bshishov
Took since I find this to be really useful code for this project and others

"""

def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(actual, predicted) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))



def normalize_data(df):
    x = df.values
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=df.columns)


def chunk_data(data, chunk_size):
    """
    function chunks the data into several equally sized data chunks, all of them meant to
    be inputted into the LSTM as such. This is a point to messing around to test performance
    to see if smaller or larger chunk sizes get better performance.

    :param data: data input, mean to be numpy array from dataFrame.values
    :param chunk_size: number of samples to take from the data given
    :return: chunked data
    """
    balancer = data.shape[0] % chunk_size
    if balancer == 0:
        return array(split(data, len(data)/chunk_size))
    else:
        # needed since all arrays must be balanced for number of features
        # This removes the extra features that would result in an uneven balance on the last array
        # only removes the final elements
        removal_list = []
        for i in range(balancer, 0, -1):
            removal_list.append(data.shape[0] - i)

        data = np.delete(data, removal_list, axis=0)
        return array(split(data, len(data) / chunk_size))


def forecast_evals(forecasted: np.ndarray, actual: np.ndarray):

    try:
        me = np.mean(forecasted - actual)  # mean error
    except Exception as err:
        me = 'Unable to compute metric {0}'.format(err)

    try:
        mae_ = np.mean(np.abs(forecasted - actual))  # mean absolute error
    except Exception as err:
        mae_ = 'Unable to compute metric {0}'.format(err)

    try:
        mse = mean_squared_error(forecasted, actual)  # mean squared error
    except Exception as err:
        mse = 'Unable to compute metric {0}'.format(err)

    try:
        rmse = np.mean((forecasted - actual) ** 2) ** .5  # Root Mean Squared Error
    except Exception as err:
        rmse = 'Unable to compute metric {0}'.format(err)

    try:
        corr = np.corrcoef(forecasted, actual)[0, 1]
    except Exception as err:
        corr = 'Unable to compute metric {0}'.format(err)

    try:
        over_expect = sum(forecasted > actual) / len(actual)  # percentage that forecasted
    except Exception as err:
        over_expect = 'Unable to compute metric {0}'.format(err)
    # over predicts the actual

    try:
        residuals = forecasted - actual
    except Exception as err:
        residuals = 'Unable to compute metric {0}'.format(err)

    try:
        mfe = np.mean(residuals)  # mean of the residuals
    except Exception as err:
        mfe = 'Unable to compute metric {0}'.format(err)

    try:
        mase_ = mase(actual, forecasted)
    except Exception as err:
        mase_ = 'Unable to compute metric {0}'.format(err)

    return residuals, ({'me': me, 'mae': mae_, "mse": mse,
              'rmse': rmse, "over_shot": over_expect, "mfe": mfe,
             'corr': corr, "mase": mase_})


def split_data_to_x_y(data, input_length, output_length=20, only_window=True, forecast_column_index=4):
    """
    Takes in a the data that has been passed through chunk_data. Reshapes it back into a numpy matrix,
    Then starts to chunk the data again. It will grab an inputted amount of lagged values that are to be used in
    the lstm. Will do the same with the forecasted value. Then return those values so that they are ready to be
    used by the lstm.

    :param forecast_column_index: index column of the what uni variate value I want to be forecasted
    :param data: my input data, will have passed through chunk_data and be in numpy arrays
    :param input_length: amount of lagged values that I want to use with each timestep
    :param output_length:
    :param only_window: controller on the type of output. If true is is univariate. If false still
    univariate but now also forecasts a series of timesteps
    :return: the x an y, info and forecast for the lstm
    """
    # put data back into continuous matrix
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    X = []
    y = []
    start_point = 0  # start gathering training data at the beginning

    # move over data one time step at a time
    for _ in range(len(data)):
        input_data_end = start_point + input_length
        # output will currently cover timestep forecasts t+1 to t+20
        output_data_end = input_data_end + output_length

        if output_data_end <= len(data):
            X.append(data[start_point:input_data_end, 0:-1])
            if only_window:
                # arrays are set to get the info into [samples, timestep, feature]
                y.append([[data[input_data_end, -1]]])
            else:
                y.append(data[input_data_end:output_data_end, forecast_column_index])

        start_point += 1  # move ahead one time step

    return array(X), array(y)


def build_model(train_x, train_y):
    # define parameters
    verbose, epochs, batch_size = 1, 10, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    # train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def forecast(model, history, n_input, single_value=True):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data, make sure to exclude forecasted value
    # important to note that grabbing the last value while appending new data is what makes this work
    input_x = data[-n_input:, 0:-1]
    expected = data[-1, -1]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week, verbose set to one for this since it's so fast
    yhat = model.predict(input_x, verbose=0)
    # get rid of a dimension
    yhat = yhat[0][0][0]
    return yhat, expected


if __name__ == "__main__":
    df_solar_train = pd.read_csv("/home/nelson/PycharmProjects/"
                           "Solar Forecasting Thesis Project/Data/"
                           "small_test_data/small_train_data.csv", index_col="time", parse_dates=True)
    df_solar_test = pd.read_csv("/home/nelson/PycharmProjects/"
                                 "Solar Forecasting Thesis Project/Data/"
                                 "small_test_data/small_test_data.csv", index_col="time", parse_dates=True)
    lagged_timesteps = 15
    data_solar_train = chunk_data(df_solar_train.values, lagged_timesteps)
    data_solar_test = chunk_data(df_solar_test.values, lagged_timesteps)
    x, y = split_data_to_x_y(data_solar_train, lagged_timesteps)
    model = build_model(x, y)
    timestep_history = [x for x in data_solar_train]
    predictions = list()
    expected_values = list()
    for i in range(len(data_solar_test)):
        predicted, expected = forecast(model, timestep_history, lagged_timesteps)
        predictions.append(predicted)
        expected_values.append(expected)
        # appends new data to the very end of the history
        # this is how the test data is added to the whole thing
        timestep_history.append(data_solar_test[i, :])

    predictions = array(predictions)
    expected_values = array(expected_values)

    residuals, scores = forecast_evals(predictions, expected_values)
    print(scores)




























