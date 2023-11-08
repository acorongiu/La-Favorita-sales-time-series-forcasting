# La-Favorita-sales-time-series-forcasting
Sales forecasting of 3 best sellers product categories for 2 (absolute best sellers) stores of la Corporacion Favorita, a large Ecuadorian-based grocery retailer.

Initial data are available on Kaggle at: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data
be careful: train file has to be added to 'data' folder in order to run the code without path isses.

# Goal: minimize MAPE (mean absolute percentage error) for 28 days (4 weeks) of sales predictions

How: comparing performance between ARIMA, Random Forest, Gradient Boosting and LSTM models and aggregating the best performing ones to generate an ensemble model that is reliable
i.e. it is robust: low variance and acceptable bias.

# Navigate the project:
data folder contains initial data stored in csv files and contains generated files that are necessary to the time series analysis and forecasting.
package contains functions.py with all the functions developed to forecast the sales allowing to try different combinations of models'paramenters.
Jupiter Notebook 'La Favorita time series analysis' shows the exploratory analysis, data pre-processing, stationarity/seasonality check and more of the time series.
Jupiter Notebook 'La Favorita predictions-no outliers' contains time series forecasting, final table of mape scores, residuals analysis to check for bias in the models.
Jupiter Notebook 'La Favorita predictions - multivariate series' contains time series forecasting of multivariate time series: each series future values are predicted based on past values of all the series,
results from each multivariate model are stored in csv files in a dedicated folder in 'data' folder.
