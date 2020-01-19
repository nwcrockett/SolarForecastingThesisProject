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


def forecast_accuracy(forecast, actual):
    """
    Taken from https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/


    :param forecast:
    :param actual:
    :return:
    """
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    # acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse, # 'acf1':acf1,
            'corr':corr, 'minmax':minmax})


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
                results = forecast_accuracy(predicted, expected)
                parameters = " p={} d={} q={}".format(p, d, q)
                print(str(results) + parameters)
                file.write(str(results) + parameters + "\n")


if __name__ == "__main__":
    df = pd.read_csv("Data/test/fully processes minute data.csv", index_col="time", parse_dates=True)
    test_arima_parameters_short_sample(df, p_max=3, d_max=2, q_max=1)









