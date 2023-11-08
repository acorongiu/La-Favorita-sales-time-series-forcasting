#!/usr/bin/env python
# coding: utf-8

#install
#pip install darts

#imports
import numpy as np
import pandas as pd
import os
import warnings

from numpy import corrcoef

import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px

from darts import TimeSeries
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller
from darts.dataprocessing.transformers import Scaler

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from darts.utils.statistics import check_seasonality

from darts.utils.timeseries_generation import datetime_attribute_timeseries

from darts.models.filtering.moving_average import MovingAverage
from darts.models import NaiveSeasonal
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.arima import ARIMA
import lightgbm
from xgboost import XGBRegressor
from darts.models.forecasting.regression_model import RegressionModel
from darts.models.forecasting.gradient_boosted_model import LightGBMModel

from darts.models import RNNModel
import torch
import torch.nn as nn
import torch.optim as optim
#import shutil
from torch.utils.tensorboard import SummaryWriter

from darts.metrics import mae
from darts.metrics import mape


#configuration
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')


#let's creat df of top series from csv
top_series = pd.read_csv('./data/top_series_train.csv', index_col='date', parse_dates=True)
top_transactions = pd.read_csv("./data/top_series_transactions.csv", index_col='date', parse_dates=True)

#replace the zeros with the next sales value
top_series = top_series.replace(to_replace=0, method='bfill')

def calculate_correlation(product: str, transactions: str, lags: int):
    """
    calculates and prints correlation between product and transactions of a store
    :param product: str
    :param transactions: str
    :param lags: int
    :return: None
    """
    lagged_trans = top_transactions[transactions].shift(lags)[lags:]
    temp = pd.merge(top_series[product], lagged_trans, left_index=True, right_index=True)
    corr = round(corrcoef(temp[product], lagged_trans)[1][0], 2)
    if corr > 0.5:
        print('correlation between {} and {} transactions with {} lags is: \n'.format(product, transactions, lags), corr)
    #return corr



def plot_autocorr(series: pd.Series, lags: int = 120):
    fig, ax = plt.subplots(figsize=(15, 5))
    plot_acf(x=series, ax=ax, lags=lags, alpha=0.05)
    name = series.name
    lags = str(lags)
    title = "{}'s autocorrelation with lag = {}".format(name, lags)
    plt.title(label=title)
    plt.tight_layout()
    plt.show()



def plot_partial_autocorr(series: pd.Series, lags: int = 90):
    fig, ax = plt.subplots(figsize=(15, 5))
    plot_pacf(x=series, ax=ax, lags=lags, alpha=0.05)
    name = series.name
    lags = str(lags)
    title = "{}'s partial autocorrelation with lag = {}".format(name, lags)
    plt.title(label=title)
    plt.tight_layout()
    plt.show()



def check_autocorrelation(series: str, diff_lags: int):
    """
    plots autocorrelation and partial autocorrelation functions of a given series
    :param series: str
    :param diff_lags: int
    :return: tuple with differenced series, list of lags for autocorrelation, list of lags for partial autocorrelation
             list of indexes for partial autocorrelation
    """
    diff_series = top_series[series].diff(diff_lags)[diff_lags:]
    
    auto_corr = acf(x=diff_series, nlags=120)
    lags_ac = [(i+1, round(c, 3)) for i, c in enumerate(auto_corr[1:]) if abs(c) > 0.4]
    
    part_corr = pacf(x=diff_series, nlags=90)
    lags_pac = [(i+1, round(c, 3)) for i, c in enumerate(part_corr[1:]) if abs(c) > 0.175] #0.2
    
    lags_indexes_pac = [int(t[0])*(-1) for t in lags_pac]
    
    plot_autocorr(series=diff_series, lags=120)
    
    plot_partial_autocorr(series=diff_series, lags=90)
    
    return diff_series, lags_ac, lags_pac, lags_indexes_pac

def check_autocorrelation_short_series(series: str):
    """
    same as check_autocorrelation() but series start from 01/01 of most recent year available
    :param series: str
    :return: tuple
    """
    series = top_series[series]

    short_series = series[-227:]

    auto_corr = acf(x=short_series, nlags=15)
    lags_ac = [(i + 1, round(c, 3)) for i, c in enumerate(auto_corr[1:]) if abs(c) > 0.6]

    part_corr = pacf(x=short_series, nlags=15)
    lags_pac = [(i + 1, round(c, 3)) for i, c in enumerate(part_corr[1:]) if abs(c) > 0.25]  # abs(c) > 0.3

    lags_indexes_pac = [int(t[0]) * (-1) for t in lags_pac]

    plot_partial_autocorr(series=short_series, lags=15)

    return short_series, lags_ac, lags_pac, lags_indexes_pac


def check_autocorrelation_last_days(series: str, diff_lags: int):
    """
    same as check_autocorrelation_short_series() but series get differenced by diff_lags
    :param series: str
    :param diff_lags: int
    :return: tuple
    """
    diff_series = top_series[series].diff(diff_lags)[diff_lags:]

    diff_series = diff_series[-227:]

    auto_corr = acf(x=diff_series, nlags=14)
    lags_ac = [(i + 1, round(c, 3)) for i, c in enumerate(auto_corr[1:]) if abs(c) > 0.6]

    part_corr = pacf(x=diff_series, nlags=14)
    lags_pac = [(i + 1, round(c, 3)) for i, c in enumerate(part_corr[1:]) if abs(c) > 0.25]  # abs(c) > 0.3

    lags_indexes_pac = [int(t[0]) * (-1) for t in lags_pac]

    plot_partial_autocorr(series=diff_series, lags=14)

    return diff_series, lags_ac, lags_pac, lags_indexes_pac



def is_series_seasonal(series: TimeSeries, lags: int = 90):
    for m in range(2, lags):
        is_seasonal, period = check_seasonality(series, m=m, max_lag=366, alpha=0.1)
        if is_seasonal:
            print("There is seasonality of order {}.".format(period))



def get_datetime_series(series: TimeSeries, month: bool = False, week: bool = True,
                        day_of_week: bool = True, days_only: bool = True, week_of_year: bool = False):
    """
    creates time series of future covariates of a given series, covariates can be included using dedicated params
    :param series:
    :param month:
    :param week:
    :param day_of_week:
    :param days_only:
    :param week_of_year:
    :return: TimeSeries
    """

    year_series = datetime_attribute_timeseries(
    pd.date_range(start=series.start_time(), freq=series.freq_str, periods=1800),
    attribute="year",
    one_hot=False)
    year_series = Scaler().fit_transform(year_series)

    if month == True:
        month_series = datetime_attribute_timeseries(
            year_series, attribute="month", one_hot=True)

    if week == True:
        week_series = datetime_attribute_timeseries(
            year_series, attribute="week", one_hot=True)

    if day_of_week == True:
        day_series = datetime_attribute_timeseries(
            year_series, attribute="dayofweek", one_hot=True)
    
    future_covariates = day_series

    # stack week and days to obtain series of 2 dimensions (week and days):
    if days_only == False and week_of_year == True:
        future_covariates = week_series.stack(day_series)
        print('settimane e giorni')
    if days_only == False and week_of_year == False:
        future_covariates = year_series.stack(day_series)
        print('anno e giorni')

    return future_covariates



def get_not_diff_predictions(pred: TimeSeries, series_name: str):
    """
    returns not differenced predictions when time series of predictions from a model is differenced
    :param pred:
    :param series_name:
    :return: TimeSeries
    """
    predictions = pred.pd_series()
    ind = predictions.index.to_list()
    
    real_pred = []
    real_pred.append(predictions[0]+top_series[series_name]['2017-07-18'])
    for i in predictions[1:]:
        lag = real_pred[-1]
        real_pred.append(round(i+lag, 2))
    
    real_pred = pd.Series(data=real_pred, index=ind, name=series_name)
    real_pred.index.rename('date', inplace=True)
    real_pred = TimeSeries.from_series(real_pred, freq='D', fill_missing_dates=True)
    
    return real_pred



def plot_predictions(series: TimeSeries, test: TimeSeries, pred: TimeSeries, model: str, lags: int = None):
    name = series.columns[0]
    fig, ax = plt.subplots(figsize=(15, 5))
    series[-84:-27].pd_series().plot(lw=4, label="train", ax=ax)
    test.pd_series().plot(lw=4, label="test", ax=ax)
    pred.pd_series().plot(lw=2, label="{} forecast".format(model), ax=ax)
    if model.lower() == 'arima':
        title = r"{} series's {} forecast with {} lags".format(name, model, str(lags))
        plt.title(title)
    else:
        title = r"{} series's {} forecast".format(name, model)
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



def measure_mae(model, test, pred):
    print("model {} obtains MAE: {:.2f}".format(model, mae(test, pred)))
    return round(mae(test, pred), 2)



def measure_mape(model, test, pred):
    print("model {} obtains MAPE: {:.2f}".format(model, mape(test, pred)))
    return round(mape(test, pred), 2)



def get_naive_predictions(series: str):
    series = TimeSeries.from_series(top_series[series], freq='D', fill_missing_dates=True)
    partial = series[-227:]  # dall'inizio dell'anno 2017

    # we create the weekly moving average series
    # ma = MovingAverage(window=7, centered=False)
    # ma_series = ma.filter(series=partial)

    # let's get the predictions
    naive_model = NaiveSeasonal(K=7)  # K=7
    # naive_model.fit(ma_series[:-28])
    naive_model.fit(partial[:-28])
    naive_forecast = naive_model.predict(28)

    # let's create the plot
    fig, ax = plt.subplots(figsize=(15, 5))
    partial[-140:].plot(lw=4, label="actual", ax=ax)
    naive_forecast.plot(lw=2, label="naive forecast (K=7)", ax=ax)
    plt.title(r'Naive Seasonal weekly moving average forecast')
    plt.tight_layout()
    plt.show()

    # let's calculate the mape
    mape = measure_mape(model='Naive Seasonal', test=partial[-28:], pred=naive_forecast)
    return mape



def get_model_predictions(diff_series: pd.Series, model_type: str, lags_indexes: list,
                          to_scale: bool = False, fut_cov_days_only: bool = True, week_of_year: bool = False,
                          lags_future_covariates: list = [-1, 0], lags_past_covariates: list = None,
                          past_covariates: TimeSeries = None, return_pred: bool = False):
    """
    Returns model mape from selected model: ARIMA class or Random Forest, if return_pred == True it returns also predictions
    :param diff_series:
    :param model_type:
    :param lags_indexes:
    :param to_scale:
    :param fut_cov_days_only:
    :param week_of_year:
    :param lags_future_covariates:
    :param lags_past_covariates:
    :param past_covariates:
    :param return_pred:
    :return:
    """

    # series's name
    name = diff_series.name
    
    # not differenced series
    original_series = TimeSeries.from_dataframe(df=top_series, value_cols=name, freq='D', fill_missing_dates=True)
    
    # creation of differenced series of darts
    series = TimeSeries.from_series(diff_series, freq='D', fill_missing_dates=True)
    
    # train test split
    train = series[:-28]
    test = series[-28:]
    
    # Data scaling, it is better to fit only on the train set
    if to_scale == True:
        scaler = Scaler()
        train = scaler.fit_transform(train)
    
    # Check seasonality
    is_series_seasonal(series=series, lags=90)
    
    # Adding future covariates
    covariates = get_datetime_series(series=series, days_only=fut_cov_days_only, week_of_year=week_of_year)
    
    # Predizione in base al tipo di modello
    if model_type.lower() == 'arima':
        p = abs(lags_indexes[-1])
        model = ARIMA(p=p, d=0, q=0)
        model.fit(train, future_covariates=covariates)
        pred = model.predict(n=28, future_covariates=covariates)
        
    if model_type.lower() == 'random forest':
        model = RandomForest(lags=lags_indexes, lags_future_covariates=lags_future_covariates,
                             lags_past_covariates=lags_past_covariates, n_estimators=500, random_state=3,
                             oob_score=True, max_features='sqrt') #n_estimators=100
        
        model.fit(train, future_covariates=covariates, past_covariates=past_covariates)
        pred = model.predict(n=28, future_covariates=covariates, past_covariates=past_covariates)
    
    # Reverse data scaling
    if to_scale == True:
        pred = scaler.inverse_transform(pred)
        
    # Not differenced predictions
    not_diff_pred = get_not_diff_predictions(pred=pred, series_name=name)
    
    # Calculating mape as the error metric
    test_series = original_series[-28:]
    model_mape = measure_mape(model=model_type, test=test_series, pred=not_diff_pred)

    # Plot
    if model_type.lower() == 'arima':
        plot_predictions(series=original_series, test=test_series, pred=not_diff_pred, model=model_type, lags=p)
    if model_type.lower() == 'random forest':    
        plot_predictions(series=original_series, test=test_series, pred=not_diff_pred, model=model_type)

    if return_pred == True:
        return not_diff_pred.pd_series()
    
    return model_mape



def get_arima_predictions(diff_series: pd.Series, best_params: list,
                          to_scale: bool = True, fut_cov_days_only: bool = True, week_of_year: bool = False,
                          lags_future_covariates: list = [-1, 0], return_pred: bool = False):
    """
    same as get_model_predictions but specific for ARIMA class
    :param diff_series:
    :param best_params:
    :param to_scale:
    :param fut_cov_days_only:
    :param week_of_year:
    :param lags_future_covariates:
    :param return_pred:
    :return:
    """
    # serie's name
    name = diff_series.name

    # not differenced serie
    original_series = TimeSeries.from_dataframe(df=top_series, value_cols=name, freq='D', fill_missing_dates=True)

    # Creation of differenced series of darts
    series = TimeSeries.from_series(diff_series, freq='D', fill_missing_dates=True)

    # train test split
    train = series[:-28]
    test = series[-28:]

    # Data scaling, it is better to fit only on the train set
    if to_scale == True:
        scaler = Scaler()
        train = scaler.fit_transform(train)

    # Check seasonality
    # is_series_seasonal(series=series, lags=90)

    # Adding future covariates
    covariates = get_datetime_series(series=series, days_only=fut_cov_days_only, week_of_year=week_of_year)

    # Prediction
    p, d, q = best_params[0], best_params[1], best_params[2]
    model = ARIMA(p=p, d=d, q=q, seasonal_order=(best_params[-4:]))
    model.fit(train, future_covariates=covariates)
    pred = model.predict(n=28, future_covariates=covariates)

    # Reverse data scaling
    if to_scale == True:
        pred = scaler.inverse_transform(pred)

    # Not differenced predictions
    not_diff_pred = get_not_diff_predictions(pred=pred, series_name=name)

    # Calculating mape as the error metric
    test_series = original_series[-28:]
    model_mape = measure_mape(model='best arima', test=test_series, pred=not_diff_pred)

    # Plot
    plot_predictions(series=original_series, test=test_series, pred=not_diff_pred, model='arima', lags=p)

    if return_pred == True:
        return not_diff_pred.pd_series()

    return model_mape



def get_rf_predictions_no_diff(series: pd.Series, lags_indexes: list, model_type: str = 'Random forest',
                               to_scale: bool = False, fut_cov_days_only: bool = True, week_of_year: bool = False,
                               lags_future_covariates: list = [-1, 0], lags_past_covariates: list = None,
                               past_covariates: TimeSeries = None, return_pred: bool = False):
    # nome serie
    name = series.name

    # Creazione serie NON differenziata di darts
    series = TimeSeries.from_series(series, freq='D', fill_missing_dates=True)

    # Divisione train e test
    train = series[:-28]
    test = series[-28:]

    # Scalatura dei dati, è meglio fare il fit solo sul train
    if to_scale == True:
        scaler = Scaler()
        train = scaler.fit_transform(train)

    # Aggiunta covariate future
    covariates = get_datetime_series(series=series, days_only=fut_cov_days_only, week_of_year=week_of_year)

    # Predizione
    # model = RandomForest(lags=lags_indexes, lags_future_covariates=lags_future_covariates,
    #                      lags_past_covariates=lags_past_covariates, n_estimators=100,
    #                      random_state=3)  # n_estimators=500
    if model_type.lower() == 'random forest':
        model = RandomForest(lags=lags_indexes, lags_future_covariates=lags_future_covariates,
                             lags_past_covariates=lags_past_covariates, n_estimators=500, random_state=3,
                             oob_score=True, max_features='sqrt')  # n_estimators=100
    if model_type.lower() == 'bagging':
        model = RandomForest(lags=lags_indexes, lags_future_covariates=lags_future_covariates,
                             lags_past_covariates=lags_past_covariates, n_estimators=300, random_state=3,
                             oob_score=True)  # n_estimators=100

    model.fit(train, future_covariates=covariates, past_covariates=past_covariates)
    pred = model.predict(n=28, future_covariates=covariates, past_covariates=past_covariates)

    # Invertire la scalatura dei dati
    if to_scale == True:
        pred = scaler.inverse_transform(pred)

    # Calcolo mape come metrica di errore
    model_mape = measure_mape(model='Random forest', test=test, pred=pred)

    # Rappresentazione grafica
    plot_predictions(series=series, test=test, pred=pred, model='Random forest')

    if return_pred == True:
        return pred.pd_series()

    return model_mape



def get_predictions_for_past_cov(diff_series: pd.Series, model_type: str, lags_indexes: list, to_scale: bool = False,
                         fut_cov_days_only: bool = True, lags_future_covariates: list = [-1, 0]):
    """"returns the predictions of the differentiated series to use as past covariates"""
    #nome serie
    name = diff_series.name
    
    #serie non differenziata
    original_series = TimeSeries.from_dataframe(df=top_series, value_cols=name, freq='D', fill_missing_dates=True)
    
    #Creazione serie differenziata di darts
    series = TimeSeries.from_series(diff_series, freq='D', fill_missing_dates=True)
    
    #Divisione train e test
    train = series[:-28]
    test = series[-28:]
    
    #Scalatura dei dati, è meglio fare il fit solo sul train
    if to_scale == True:
        scaler = Scaler()
        train = scaler.fit_transform(train)

    #Verifica stagionalità
    is_series_seasonal(series=series, lags=90)
    
    #Aggiunta covariate future
    fut_cov = get_datetime_series(series=series, days_only=fut_cov_days_only)
    
    #Predizione in base al tipo di modello
    if model_type.lower() == 'arima':
        p = abs(lags_indexes[-1])
        model = ARIMA(p=p, d=0, q=0)
        model.fit(train, future_covariates=fut_cov)
        pred = model.predict(n=28, future_covariates=fut_cov)
        
    if model_type.lower() == 'random forest':
        model = RandomForest(lags=lags_indexes, lags_future_covariates=lags_future_covariates, n_estimators=500, random_state=3)
        model.fit(train, future_covariates=fut_cov)
        pred = model.predict(n=28, future_covariates=fut_cov)
    
    #Invertire la scalatura dei dati
    if to_scale == True:
        pred = scaler.inverse_transform(pred)
        
    #Predizioni non differenziate
    #not_diff_pred = get_not_diff_predictions(pred=pred, series_name=name)
    
    return pred #not_diff_pred



def get_rnn_predictions_no_diff(series: TimeSeries, model_type: str,
                        to_scale: bool = True, fut_cov_days_only: bool = True,
                        week_of_year: bool = False,
                        past_covariates: TimeSeries = None):
    """
    To be used when you want to predict the NOT-differentiated series with recurrent neural network
    """

    #nome serie

    #Divisione train e test
    train = series[:-28]
    test = series[-28:]
    
    #Scalatura dei dati, è meglio fare il fit solo sul train
    if to_scale == True:
        scaler = Scaler()
        train = scaler.fit_transform(train)

    #Verifica stagionalità
    is_series_seasonal(series=series, lags=15)
    
    #Aggiunta covariate future
    fut_cov = get_datetime_series(series=series, days_only=fut_cov_days_only, week_of_year=week_of_year)
    
    if model_type.lower() == 'lstm':
        rnn = RNNModel(model="LSTM",
                        hidden_dim=30, #25
                        n_rnn_layers=1,
                        dropout=0,
                        batch_size=8, #16
                        n_epochs=40, #100
                        loss_fn=nn.MSELoss(),
                        optimizer_kwargs={"lr": 1e-3},
                        random_state=42,
                        #training_length=30, #15
                        input_chunk_length=28, #14
                        output_chunk_length=1,
                      )
        
        rnn.fit(train, future_covariates=fut_cov, max_samples_per_ts=82,  verbose=True) #max_samples_per_ts=100
        pred = rnn.predict(n=28, future_covariates=fut_cov)
    
    #Invertire la scalatura dei dati
    if to_scale == True:
        pred = scaler.inverse_transform(pred)

    #Calcolo mape come metrica di errore
    model_mape = measure_mape(model=model_type, test=test, pred=pred)

    #Rappresentazione grafica
    plot_predictions(series=series, test=test, pred=pred, model='Lstm')
    
    return model_mape



def get_rnn_predictions_diff(diff_series: pd.Series, model_type: str, lags_indexes: list,
                        to_scale: bool = True, fut_cov_days_only: bool = True,
                        week_of_year: bool = False, return_pred: bool = False,
                        past_covariates: TimeSeries = None):
    """
    To be used when you want to predict the differentiated series with recurrent neural network
    :param diff_series:
    :param model_type:
    :param lags_indexes:
    :param to_scale:
    :param fut_cov_days_only:
    :param week_of_year:
    :param return_pred:
    :param past_covariates:
    :return:
    """

    #nome serie
    name = diff_series.name
    
    #Creazione ts darts
    series = TimeSeries.from_series(diff_series, freq='D', fill_missing_dates=True)
    
    #serie non differenziata
    original_series = TimeSeries.from_dataframe(df=top_series, value_cols=name, freq='D', fill_missing_dates=True)
    
    #Divisione train e test
    train = series[:-28]
    test = series[-28:]
    
    #Scalatura dei dati, è meglio fare il fit solo sul train
    if to_scale == True:
        scaler = Scaler()
        train = scaler.fit_transform(train)
    
    #Verifica stagionalità
    is_series_seasonal(series=series, lags=90)
    
    #Aggiunta covariate future
    fut_cov = get_datetime_series(series=series, days_only=fut_cov_days_only, week_of_year=week_of_year)
    
    #Previsione RNN
    chunk_lenght = abs(lags_indexes[-1])
    
    model_type = model_type.upper()
    rnn = RNNModel(model=model_type,
                    hidden_dim=20, #25
                    n_rnn_layers=1,
                    dropout=0.025, #0.1
                    batch_size=8, #16
                    n_epochs=60, #30
                    loss_fn=nn.MSELoss(),
                    optimizer_kwargs={"lr": 1e-3},
                    random_state=42,
                    #training_length=chunk_lenght+2,
                    input_chunk_length=chunk_lenght,
                    output_chunk_length=1
                  )
    
    #consideriamo solo gli ultimi 80 samples circa per il training
    max_samples = int(((len(train) +1) - chunk_lenght) / 20)
    
    #fit e predict
    rnn.fit(train, future_covariates=fut_cov, max_samples_per_ts=max_samples, verbose=True)
    pred = rnn.predict(n=28, future_covariates=fut_cov)
    
    #Invertire la scalatura dei dati
    if to_scale == True:
        pred = scaler.inverse_transform(pred)
    
    #Predizioni non differenziate
    pred = get_not_diff_predictions(pred=pred, series_name=name)
    
    #Calcolo mape come metrica di errore
    test = original_series[-28:]
    model_mape = measure_mape(model=model_type, test=test, pred=pred)

    #Rappresentazione grafica
    plot_predictions(series=original_series, test=test, pred=pred, model='Lstm')

    if return_pred == True:
        return pred.pd_series()
    
    return model_mape



def get_rnn_predictions_val(diff_series: pd.Series, lags_indexes: list,
                            model_type: str = 'lstm', to_scale: bool = True, fut_cov_days_only: bool = True,
                            week_of_year: bool = False, return_pred: bool = False,
                            past_covariates: TimeSeries = None):
    """
    Optimize RNN parameters using validation set
    :param diff_series:
    :param lags_indexes:
    :param model_type:
    :param to_scale:
    :param fut_cov_days_only:
    :param week_of_year:
    :param return_pred:
    :param past_covariates:
    :return:
    """
    # nome serie
    name = diff_series.name

    # Creazione ts darts
    series = TimeSeries.from_series(diff_series, freq='D', fill_missing_dates=True)

    # serie non differenziata
    original_series = TimeSeries.from_dataframe(df=top_series, value_cols=name, freq='D', fill_missing_dates=True)

    # Divisione train e test
    train = series[-392:-28]  # un anno di dati = 52 settimane; /4 per trovare max_samples
    # train = series[196:-28] #mezzo anno di dati
    # train = series[:-28]
    test = series[-28:]
    validation = series[-42:-14]  # un mese di validation #-42

    # Scalatura dei dati, è meglio fare il fit solo sul train
    if to_scale == True:
        scaler = Scaler()
        train = scaler.fit_transform(train)
        validation = scaler.transform(validation)

    # Aggiunta covariate future
    fut_cov = get_datetime_series(series=series, days_only=fut_cov_days_only, week_of_year=week_of_year)

    # Previsione RNN
    chunk_lenght = abs(lags_indexes[-1])
    # chunk_lenght = 7
    # chunk_lenght = 28
    print(chunk_lenght)

    model_type = model_type.upper()
    rnn = RNNModel(model=model_type,
                   hidden_dim=20,  # 25
                   n_rnn_layers=1,  # 2
                   dropout=0.025,  # 0.1
                   batch_size=8,  # 16
                   n_epochs=60,  # 30
                   # loss_fn=nn.L1Loss(),
                   loss_fn=nn.MSELoss(),
                   optimizer_kwargs={"lr": 1e-3},
                   random_state=42,
                   input_chunk_length=chunk_lenght,
                   output_chunk_length=1,
                   model_name="Sales_RNN",
                   log_tensorboard=True,
                   force_reset=True,
                   save_checkpoints=True
                   )

    # consideriamo solo gli ultimi 80 samples circa per il training
    max_samples = int(((len(train) + 1) - chunk_lenght) / 4)  # 20 se train = series[:-28]
    print(max_samples)

    # fit e predict
    rnn.fit(train, future_covariates=fut_cov, max_samples_per_ts=max_samples,
            val_series=validation, val_future_covariates=fut_cov, verbose=True)

    # Use the best model obtained over training, according to validation loss:
    best_model = RNNModel.load_from_checkpoint(model_name="Sales_RNN", best=True)

    pred = best_model.predict(n=28, future_covariates=fut_cov)

    # Invertire la scalatura dei dati
    if to_scale == True:
        pred = scaler.inverse_transform(pred)

    # Predizioni non differenziate
    pred = get_not_diff_predictions(pred=pred, series_name=name)

    # Calcolo mape come metrica di errore
    test = original_series[-28:]
    model_mape = measure_mape(model=model_type, test=test, pred=pred)

    # Rappresentazione grafica
    plot_predictions(series=original_series, test=test, pred=pred, model='Lstm')

    if return_pred == True:
        return pred.pd_series()

    return model_mape



def get_xgboost_predictions(series: pd.Series, lags_indexes: list,
                            model_type: str, to_scale: bool = False,
                            fut_cov_days_only: bool = True, week_of_year: bool = False,
                            lags_future_covariates: list = [-1, 0], lags_past_covariates: list = None,
                            past_covariates: TimeSeries = None, return_pred: bool = False):
    """
    Returns mape from predictions by xgboost
    :param series:
    :param lags_indexes:
    :param model_type:
    :param to_scale:
    :param fut_cov_days_only:
    :param week_of_year:
    :param lags_future_covariates:
    :param lags_past_covariates:
    :param past_covariates:
    :param return_pred:
    :return:
    """
    # nome serie
    name = series.name

    # media vendite dall'inizio dell'anno
    #last_year_to_date_mean = series['02-01-2017':'18-07-2017'].mean()
    last_two_weeks_mean = series['05-07-2017':'18-07-2017'].mean()

    # Creazione serie differenziata di darts
    series = TimeSeries.from_series(series, freq='D', fill_missing_dates=True)
    # series = series[-730:]

    # Divisione train e test
    train = series[:-28]
    test = series[-28:]
    #validation = series[-28:-13]
    validation = series[-42:-14] #42=14+28 per usare fino a lag 28

    # Scalatura dei dati, è meglio fare il fit solo sul train
    if to_scale == True:
        scaler = Scaler()
        train = scaler.fit_transform(train)
        validation = scaler.transform(validation)

    # Aggiunta covariate future
    covariates = get_datetime_series(series=series, days_only=fut_cov_days_only, week_of_year=week_of_year)

    # Predizione
    early_stopping = lightgbm.early_stopping(stopping_rounds=300, first_metric_only=False, verbose=False)

    if model_type.lower() == 'xgboost':
        model = RegressionModel(lags=lags_indexes, lags_past_covariates=lags_past_covariates,
                                lags_future_covariates=lags_future_covariates,
                                output_chunk_length=1, model=XGBRegressor(n_estimators=500, max_depth=None,
                                                                          max_leaves=31, max_bin=1000,
                                                                          learning_rate=0.005,
                                                                          base_score=last_two_weeks_mean))
        model.fit(train, future_covariates=covariates)

    if model_type.lower() == 'lightgboost':
        model = LightGBMModel(lags=lags_indexes, lags_past_covariates=lags_past_covariates,
                              lags_future_covariates=lags_future_covariates,
                              output_chunk_length=1, random_state=42,
                              num_leaves=5, learning_rate=0.005, n_estimators=850) #num_leaves=31, n_estimators=500
        model.fit(train, future_covariates=covariates, val_series=validation, val_future_covariates=covariates,
                  verbose=False, callbacks=[early_stopping])
    pred = model.predict(n=28, future_covariates=covariates)

    # Invertire la scalatura dei dati
    if to_scale == True:
        pred = scaler.inverse_transform(pred)

    # Calcolo mape come metrica di errore
    model_mape = measure_mape(model='Xgboost', test=test, pred=pred)

    # Rappresentazione grafica
    plot_predictions(series=series, test=test, pred=pred, model=model_type)

    if return_pred == True:
        return pred.pd_series()

    return model_mape



def get_xgboost_predictions_diff(diff_series: pd.Series, lags_indexes: list,
                                 model_type: str, to_scale: bool = False,
                                 fut_cov_days_only: bool = True, week_of_year: bool = False,
                                 lags_future_covariates: list = [-1, 0], lags_past_covariates: list = None,
                                 past_covariates: TimeSeries = None, return_pred: bool = False):
    # nome serie
    name = diff_series.name

    # serie non differenziata
    original_series = TimeSeries.from_dataframe(df=top_series, value_cols=name, freq='D', fill_missing_dates=True)

    # Creazione serie differenziata di darts
    series = TimeSeries.from_series(diff_series, freq='D', fill_missing_dates=True)

    # Divisione train e test
    train = series[:-28]
    test = series[-28:]
    #validation = series[-28:-13]
    validation = series[-42:-14]

    # Scalatura dei dati, è meglio fare il fit solo sul train
    if to_scale == True:
        scaler = Scaler()
        train = scaler.fit_transform(train)
        validation = scaler.transform(validation)

    # Verifica stagionalità
    # is_series_seasonal(series=series, lags=90)

    # Aggiunta covariate future
    covariates = get_datetime_series(series=series, days_only=fut_cov_days_only, week_of_year=week_of_year)

    # Predizione
    # early_stopping = lightgbm.early_stopping(stopping_rounds=30, first_metric_only=False, verbose=False)

    if model_type.lower() == 'xgboost':
        model = RegressionModel(lags=lags_indexes, lags_past_covariates=lags_past_covariates,
                                lags_future_covariates=lags_future_covariates,
                                output_chunk_length=1, model=XGBRegressor(n_estimators=500, max_depth=None,
                                                                          max_leaves=31, max_bin=1000,
                                                                          learning_rate=0.008,
                                                                          base_score=diff_series[:-28].mean()))
        model.fit(train, future_covariates=covariates, )  # eval_set=validation)
    if model_type.lower() == 'lightgboost':
        model = LightGBMModel(lags=lags_indexes, lags_past_covariates=lags_past_covariates,
                              lags_future_covariates=lags_future_covariates,
                              output_chunk_length=1, random_state=42,
                              num_leaves=31, learning_rate=0.005, n_estimators=500, max_bin=1000)
        model.fit(train, future_covariates=covariates, val_series=validation, val_future_covariates=covariates,
                  verbose=False)  # callbacks=[early_stopping])
    pred = model.predict(n=28, future_covariates=covariates)  # validate_features=True)

    # Invertire la scalatura dei dati
    if to_scale == True:
        pred = scaler.inverse_transform(pred)

    # Predizioni non differenziate
    not_diff_pred = get_not_diff_predictions(pred=pred, series_name=name)

    # Calcolo mape come metrica di errore
    test_series = original_series[-28:]
    model_mape = measure_mape(model='Xgboost', test=test_series, pred=not_diff_pred)

    # Rappresentazione grafica
    plot_predictions(series=original_series, test=test_series, pred=not_diff_pred, model='Xgboost')

    if return_pred == True:
        return not_diff_pred.pd_series()

    return model_mape