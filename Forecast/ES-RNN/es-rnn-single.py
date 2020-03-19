import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, sampler, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from numpy import array
from numpy import split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

EPSILON = 1e-10
np.set_printoptions(precision=3, suppress=True)
TEST_DATA_PATH = "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/test/"
RESULTS_STORAGE_PATH = "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Forecast/ARIMA/"

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


"""
This starts off a marked off section of code taken from another source cited below
Taken from https://github.com/lysecret2/ES-RNN-Pytorch
author: Slawek Smyl
Date: 189MAR2020

Code was taken since I'm using Slawek's ES-RNN algorithm to forecast
"""

class holt_winters_no_trend(torch.nn.Module):

    def __init__(self, init_a=0.1, init_g=0.1, slen=12):

        super(holt_winters_no_trend, self).__init__()

        # Smoothing parameters
        self.alpha = torch.nn.Parameter(torch.tensor(init_a))
        self.gamma = torch.nn.Parameter(torch.tensor(init_g))

        # init parameters
        self.init_season = torch.nn.Parameter(torch.tensor(np.random.random(size=slen)))

        # season legnth used to pick appropriate past season step
        self.slen = slen

        # Sigmoid used to norm the params to be betweeen 0 and 1 if needed
        self.sig = nn.Sigmoid()

    def forward(self, series, series_shifts, n_preds=12, rv=False):

        # Get Batch size
        batch_size = series.shape[0]

        # Get the initial seasonality parameter
        init_season_batch = self.init_season.repeat(batch_size).view(batch_size, -1)

        # We use roll to Allow for our random input shifts.
        seasonals = torch.stack([torch.roll(j, int(rol)) for j, rol in zip(init_season_batch, series_shifts)]).float()

        # It has to be a list such that we dont need inplace tensor changes.
        seasonals = list(torch.split(seasonals, 1, dim=1))
        seasonals = [x.squeeze() for x in seasonals]

        # Now We loop over the input in each forward step
        result = []

        # rv can be used for decomposing a series./normalizing in case of ES-RNN
        if rv == True:
            value_list = []
            season_list = []

        for i in range(series.shape[1] + n_preds):

            # 0th step we init the parameter
            if i == 0:
                smooth = series[:, 0]
                value_list.append(smooth)
                season_list.append(seasonals[i % self.slen])
                result.append(series[:, 0])

                continue

            # smoothing
            # its smaller here, so smoothing is only for one less than the input?
            if i < series.shape[1]:

                val = series[:, i]

                last_smooth, smooth = smooth, self.sig(self.alpha) * (val - seasonals[i % self.slen]) + (
                        1 - self.sig(self.alpha)) * (smooth)

                seasonals[i % self.slen] = self.sig(self.gamma) * (val - smooth) + (1 - self.sig(self.gamma)) * \
                                           seasonals[i % self.slen]

                # we store values, used for normaizing in ES RNN
                if rv == True:
                    value_list.append(smooth)
                    season_list.append(seasonals[i % self.slen])

                result.append(smooth + seasonals[i % self.slen])

            # forecasting would jsut select last smoothed value and the appropriate seasonal, we will do this seperately
            # in the ES RNN implementation
            else:

                m = i - series.shape[1] + 1

                result.append((smooth) + seasonals[i % self.slen])

                if rv == True:
                    value_list.append(smooth)
                    season_list.append(seasonals[i % self.slen])

            # If we want to return the actual, smoothed values or only the forecast
        if rv == False:
            return torch.stack(result, dim=1)[:, -n_preds:]
        else:
            return torch.stack(result, dim=1), torch.stack(value_list, dim=1), torch.stack(season_list, dim=1)


class es_rnn(torch.nn.Module):

    def __init__(self, hidden_size=16, slen=12, pred_len=12):
        super(es_rnn, self).__init__()

        self.hw = holt_winters_no_trend(init_a=0.1, init_g=0.1)
        self.RNN = nn.GRU(hidden_size=hidden_size, input_size=1, batch_first=True)
        self.lin = nn.Linear(hidden_size, pred_len)
        self.pred_len = pred_len
        self.slen = slen

    def forward(self, series, shifts):
        # Get Batch size
        batch_size = series.shape[0]
        result, smoothed_value, smoothed_season = self.hw(series, shifts, rv=True, n_preds=0)

        de_season = series - smoothed_season
        de_level = de_season - smoothed_value
        noise = torch.randn(de_level.shape[0], de_level.shape[1])
        noisy = de_level  # +noise
        noisy = noisy.unsqueeze(2)
        # noisy=noisy.permute(1,0,2)
        # take the last element in the sequence t agg (can also use attn)
        feature = self.RNN(noisy)[1].squeeze()  # [-1,:,:]
        pred = self.lin(feature)

        # äthe season forecast entail just taking the correct smooothed values
        season_forecast = []
        for i in range(self.pred_len):
            season_forecast.append(smoothed_season[:, i % self.slen])
        season_forecast = torch.stack(season_forecast, dim=1)

        # in the end we multiply it all together and we are done!
        # here additive seems to work a bit better, need to make that an if/else of the model
        return smoothed_value[:, -1].unsqueeze(1) + season_forecast + pred


class sequence_labeling_dataset(Dataset):

    def __init__(self, input, max_size=100, sequence_labeling=True, seasonality=12, out_preds=12):

        self.data = input
        self.max_size = max_size
        self.sequence_labeling = sequence_labeling
        self.seasonality = seasonality
        self.out_preds = out_preds

    def __len__(self):

        return int(10000)

    def __getitem__(self, index):

        data_i = self.data

        # we randomly shift the inputs to create more data
        if len(data_i) > self.max_size:
            max_rand_int = len(data_i) - self.max_size
            # take a random start integer
            start_int = random.randint(0, max_rand_int)
            data_i = data_i[start_int:(start_int + self.max_size)]
        else:
            start_int = 0

        inp = np.array(data_i[:-self.out_preds])

        if self.sequence_labeling == True:
            # in case of sequence labeling, we shift the input by the range to output
            out = np.array(data_i[self.out_preds:])
        else:
            # in case of sequnec classification we return only the last n elements we
            # need in the forecast
            out = np.array(data_i[-self.out_preds:])

        # This defines, how much we have to shift the season
        shift_steps = start_int % self.seasonality

        return inp, out, shift_steps


def run_es_rnn(seq):
    train = seq[:-20]
    test = seq

    sl = sequence_labeling_dataset(train, 1000, False, 20, 20)
    sl_t = sequence_labeling_dataset(test, 1000, False, 20, 20)

    train_dl = DataLoader(dataset=sl,
                          batch_size=512,
                          shuffle=False)

    test_dl = DataLoader(dataset=sl_t,
                         batch_size=512,
                         shuffle=False)

    hw = es_rnn(slen=20, pred_len=20)
    opti = torch.optim.Adam(hw.parameters(), lr=0.01)

    overall_loss = []
    batch = next(iter(test_dl))
    inp = batch[0].float()  # .unsqueeze(2)
    out = batch[1].float()  # .unsqueeze(2).float()
    shifts = batch[2].numpy()
    pred = hw(inp, shifts)

    overall_loss_train = []
    overall_loss = []
    for j in tqdm(range(5)):
        loss_list_b = []
        train_loss_list_b = []
        # here we use batches of past, and to be forecasted value
        # batches are determined by a random start integer
        for batch in iter(train_dl):
            opti.zero_grad()
            inp = batch[0].float()  # .unsqueeze(2)
            out = batch[1].float()  # .unsqueeze(2).float()
            shifts = batch[2].numpy()
            # it returns the whole sequence atm
            print(inp)
            print()
            print(shifts)
            pred = hw(inp, shifts)
            loss = (torch.mean((pred - out) ** 2)) ** (1 / 2)
            train_loss_list_b.append(loss.detach().cpu().numpy())

            loss.backward()
            opti.step()

        # here we use all the available values to forecast the future ones and eval on it
        for batch in iter(test_dl):
            inp = batch[0].float()  # .unsqueeze(2)
            out = batch[1].float()  # .unsqueeze(2).float()
            shifts = batch[2].numpy()
            pred = hw(inp, shifts)
            # loss=torch.mean(torch.abs(pred-out))
            loss = (torch.mean((pred - out) ** 2)) ** (1 / 2)
            loss_list_b.append(loss.detach().cpu().numpy())

        print(np.mean(loss_list_b))
        print(np.mean(train_loss_list_b))
        overall_loss.append(np.mean(loss_list_b))
        overall_loss_train.append(np.mean(train_loss_list_b))

    batch = next(iter(test_dl))
    inp = batch[0].float()  # .unsqueeze(2)
    out = batch[1].float()  # .unsqueeze(2).float()
    shifts = batch[2].numpy()
    pred = hw(torch.cat([inp, out], dim=1), shifts)

    return float(out[-1][-1]), float(pred[-1][-1])

"""
This ends a marked off section of code taken from another source cited below
Taken from https://github.com/lysecret2/ES-RNN-Pytorch
author: Slawek Smyl
Date: 189MAR2020

Code was taken since I'm using Slawek's ES-RNN algorithm to forecast
"""


def setup_data(data, cut_size=60):
    hour = []
    count = 0

    for i in data["downwelling_shortwave"].values:
        if count + cut_size >= df_solar_test["downwelling_shortwave"].values.shape[0]:
            break
        cut = df_solar_test["downwelling_shortwave"].values[count:count + cut_size]
        count += 1
        hour.append(cut)

    return np.array(hour)


if __name__ == "__main__":
    testing_data = "/home/nelson/PycharmProjects/" \
                   "Solar Forecasting Thesis Project/Data/" \
                   "test/fully processes minute data.csv"

    df_solar_test = pd.read_csv(testing_data, index_col="time", parse_dates=True)

    df_solar_test = df_solar_test["2013-07"]

    hours = setup_data(df_solar_test)

    actual = []
    forecast = []
    count_length = 0
    index = []

    for seq in hours[0:20]:
        index.append(count_length)
        print("{} current count. {} need to reach".format(count_length, len(hours)))
        a, f = run_es_rnn(seq)
        actual.append(a)
        forecast.append(f)

        count_length += 1

    actual = np.array(actual)
    forecast = np.array(forecast)

    d = {"index": index, "expected": actual, "predicted": forecast}
    df_storage = pd.DataFrame(data=d)
    df_storage.to_csv("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Forecast/ES-RNN/ESRNN-results.csv")

    file = open("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Forecast/ES-RNN/ESRNN-results.txt", "w")

    residuals, results = forecast_evals(forecast, actual)
    print(results)
    file.write(str(results))











