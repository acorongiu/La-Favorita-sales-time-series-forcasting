# La-Favorita-sales-time-series-forcasting
Sales forecasting of 3 best sellers product categories for 2 (absolute best sellers) stores of la Corporacion Favorita, a large Ecuadorian-based grocery retailer.

Initial data are available on Kaggle at: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data

# Goal: minimize MAPE (mean absolute percentage error) for 28 days (one month) of sales predictions

How: comparing performance between ARIMA, Random Forest, Gradient Boosting and LSTM models and aggregating the best performing ones to generate an ensemble model that is reliable
i.e. it is robust: low variance and acceptable bias.

# Navigate the project:
data folder contains initial data stored in csv files and contains generated files that are necessary during the time series analysis and forecasting.
package contains functions.py with all the functions developed to forecast the sales trying different combinations of models'paramenters.
Jupiter Notebook 'La Favorita time series analysis' shows the exploratory analysis, data pre-processing, stationarity/seasonality check and more of the time series
Jupiter Notebook 'La Favorita predictions-no outliers' contains time series forecasting, table of mape scores, residuals analysis to check for bias in the models.
