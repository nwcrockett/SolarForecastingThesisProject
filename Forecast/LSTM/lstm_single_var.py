"""
Nelson Crockett
19JAN20
LSTM multi-variable code for solar forecasting thesis project

I must note that I have code from github user Boris Shishov bshishov
Taken from https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
Used to evaluate forecast variables. Mainly for Mean absolute scaled error

I also had part of my code inspired from code from Jason Brownlee
book Deep Learning Time Series Forecasting
publisher machine learning mastery


"""


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from numpy import array
from numpy import split
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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


def forecast_evals(forecasted: np.ndarray, actual: np.ndarray):
    """
    Calculates several different forecast evaluation methods.

    :param forecasted: data predicted using the forecast model
    :param actual: actual value excepted
    :return:
    """

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


def forecast(model, history, n_input):
    """
    Takes in the lstm model, data, and the lagged input value. Flattens out the data into a 2D matrix
    then grabs the amount of time steps based on the lag value. Excluding the forecast data.
    Reshapes that data into the expected [samples, timesteps, features] for the forecast. For this
    the sample will be one. Makes the forecast while grabbing the expected value from the data for a
    20 minute forecast. Returns both of those values with only the numbers no added dimensions

    :param model: trained LSTM model
    :param history: starts off with all training data, then adds in new testing data
    :param n_input: size of the lagged input features
    :return: forecasted value and expected value
    """
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data, make sure to exclude forecasted value
    # important to note that grabbing the last value while appending new data is what makes this work
    input_x = data[-n_input:, 4]
    expected = data[-1, -1]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week, verbose set to one for this since it's so fast
    yhat = model.predict(input_x, verbose=0)
    # get rid of a dimension
    yhat = yhat[0][0]
    return yhat, expected


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
    :return: the x_train an y_train, info and forecast for the lstm
    """
    # put data back into continuous matrix
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    x_train = []
    y_train = []
    start_point = 0  # start gathering training data at the beginning

    # move over data one time step at a time
    for _ in range(len(data)):
        input_data_end = start_point + input_length
        # output will currently cover timestep forecasts t+1 to t+20
        output_data_end = input_data_end + output_length

        if output_data_end <= len(data):
            x_in = data[start_point:input_data_end, 4]
            x_train.append(x_in.reshape((len(x_in), 1)))
            if only_window:
                # arrays are set to get the info into [samples, timestep, feature]
                y_train.append([data[input_data_end, -1]])
            else:
                y_train.append(data[input_data_end:output_data_end, forecast_column_index])

        start_point += 1  # move ahead one time step

    return array(x_train), array(y_train)


def build_model(lag, train_x, train_y, val=None, epochs=10, verbose=1, batch_size=16):
    """
    Builds and fits the LSTM model. Will save the best current model as training occurs.
    Also will currently stop early if a better model is not found within 5 epochs.

    :param lag: amount of lag used in the model for timesteps. Only in here to add to the saved file to
    differenate between different LSTM models. So that different lag values can be tested.
    :param train_x: features that will be used to train the lstm. Multi variable in this code
    :param train_y: expected output
    :param val: validation data
    :param epochs: number of epochs to be used to train the model
    :param verbose: controller for keras if output should be given as the model is trained. Set to true
    :param batch_size: Batch size for the training data.
    :return: the trained model and the model history.
    """
    # define parameters
    lag = str(lag)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    # train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    filepath = "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project" \
                          "/Forecast/LSTM/LSTM_single_var_lag_" + lag + \
                          "-weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5."
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callbacks_list = [checkpoint, es]
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
              verbose=verbose, validation_data=val, callbacks=callbacks_list)
    return model, history


def make_lstm(lagged_timesteps, epochs):
    """
        Makes, evaluates, and graphs the loss of an LSTM

        :param lagged_timesteps: amount of lag for the time steps
        :param epochs: number of epochs to use
        :return: nothing. Will output save LSTM models, txt file with saved eval scores, and a png of model loss
        """

    # make sure to absolute paths since errors can occur if only relative paths are used

    training_data = "/home/nelson/PycharmProjects/" \
                    "Solar Forecasting Thesis Project/Data/" \
                    "train/fully processes minute data.csv"
    testing_data = "/home/nelson/PycharmProjects/" \
                   "Solar Forecasting Thesis Project/Data/" \
                   "test/fully processes minute data.csv"
    validation_data = "/home/nelson/PycharmProjects/" \
                      "Solar Forecasting Thesis Project/" \
                      "Data/validate/fully processes minute data.csv"

    score_file = "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/" \
                 "Forecast/LSTM/LSTM_scores_single_var_lag_{0}.txt".format(lagged_timesteps)
    loss_figure_file = "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/" \
                       "Forecast/LSTM/LSTM_single_var-loss-lag-{0}.png".format(lagged_timesteps)

    df_solar_train = pd.read_csv(training_data, index_col="time", parse_dates=True)
    df_solar_test = pd.read_csv(testing_data, index_col="time", parse_dates=True)
    df_solar_val = pd.read_csv(validation_data, index_col="time", parse_dates=True)

    data_solar_train = chunk_data(df_solar_train.values, lagged_timesteps)
    data_solar_test = chunk_data(df_solar_test.values, lagged_timesteps)
    data_solar_val = chunk_data(df_solar_val.values, lagged_timesteps)

    x_val, y_val = split_data_to_x_y(data_solar_val, lagged_timesteps)
    x_train, y_train = split_data_to_x_y(data_solar_train, lagged_timesteps)

    model, history = build_model(lagged_timesteps,
                                 x_train, y_train, val=(x_val, y_val), epochs=epochs)
    timestep_history = list(data_solar_train[-2:])

    # began forecasting on the testing data
    predictions = list()
    expected_values = list()

    for i in range(len(data_solar_test)):
        predicted, expected = forecast(model, timestep_history, lagged_timesteps)
        predictions.append(predicted)
        expected_values.append(expected)
        # appends new data to the very end of the history
        # this is how the test data is added to the whole thing
        timestep_history.append(data_solar_test[i, :])
        timestep_history = timestep_history[-2:]

    predictions = array(predictions)
    expected_values = array(expected_values)

    # calculate metrics for eval
    residuals_6_months, scores_6_months = forecast_evals(predictions[:int(len(predictions) / 4)],
                                                         expected_values[:int(len(predictions) / 4)])
    file = open(score_file, "w")
    file.write(str(scores_6_months) + " 1/4 scores\n")
    print(str(scores_6_months) + " 1/4 scores\n")

    residuals_year, scores_year = forecast_evals(predictions[:int(len(predictions) / 2)],
                                                 expected_values[:int(len(predictions) / 2)])
    file.write(str(scores_year) + " 1/2 scores\n")
    print(str(scores_year) + " 1/2 scores\n")

    residuals, scores = forecast_evals(predictions, expected_values)
    file.write(str(scores) + " full scores\n")
    print(str(scores) + " full scores\n")

    # graph loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(history.history["loss"])), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, len(history.history["val_loss"])), history.history["val_loss"], label="val_loss")
    plt.title("Validation loss and training loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_figure_file)
    plt.show()


if __name__ == "__main__":

    e = 50
    lags = [20, 30, 40]

    for i in lags:
        make_lstm(i, e)






























