import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.stattools import acf
import pmdarima as pm

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


def long_form_arima_test_exploration():
    """
    This function will test several different parameters of the arima forecasting.
    This will identify which set of parameters is optimal for using ARIMA on the solar
    dataset.

    Right now this function is only used to to establish that a data wide training
    for forecasting is both way too long and does not perform with a high rate of success.
    :return: Nothing
    """
    df = pd.read_csv("Data/train/fully processing minute data.csv", index_col="time", parse_dates=True)

    file = open("Arima_output.txt", "w")

    model = pm.auto_arima(df["downwelling_shortwave"].values, start_p=0, start_q=0,
                          test='adf',  # use adftest to find optimal 'd'
                          max_p=10, max_q=3,  # maximum p and q
                          m=1,  # frequency of series
                          d=None,  # let model determine 'd'
                          seasonal=False,  # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    print(model.summary())
    file.write(model.summary())


def arima_short_range_forecast(df, sample_range=20, p=2, d=1, q=0):
    # need to cover all zero values failure
    predicted = []
    expected = []
    for i in range(0, len(df)):
        if i + sample_range > len(df):
            break
        sample = df[i:i + sample_range]  # sample
        check_vals = sample["downwelling_shortwave"].values
        if check_vals.all() == 0:  # all 0 values cannot be forecasted with ARIMA
            continue

        model = ARIMA(sample["downwelling_shortwave"], order=(p, d, q))
        model_fit = model.fit(disp=0)
        forecast = model_fit.forecast(steps=20)
        predicted.append(forecast[0][-1])
        expected.append(sample["forecast_downwelling_shortwave"][-1])
    return predicted, expected


def test_arima_parameters_short_sample(df, p_max=10, d_max=2, q_max=3, start_range="2015-04", end_range="2015-06"):
    sample = df[start_range:end_range]
    file = open("Forecast/ARIMA/arima_parameter_test_short.txt", "w")
    for d in range(1, d_max + 1):
        for q in range(q_max + 1):
            for p in range(p_max + 1):
                predicted, expected = arima_short_range_forecast(sample, p=p, d=d, q=q)
                predicted = np.array(predicted)
                expected = np.array(expected)
                results = forecast_evals(predicted, expected)
                parameters = " p={} d={} q={}".format(p, d, q)
                print(str(results) + parameters)
                file.write(str(results) + parameters + "\n")


if __name__ == "__main__":
    df = pd.read_csv("Data/test/fully processes minute data.csv", index_col="time", parse_dates=True)
    test_arima_parameters_short_sample(df, p_max=3, d_max=2, q_max=1)









