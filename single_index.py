
import pandas as pd
import datetime as dt
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from pandas_datareader import data
from ticker import *
from portfolio import *


# stocks = ["SPY"]
# start = dt.datetime(2016,1,1)
# end = dt.datetime(2017,12,31)
start = dt.datetime(2016,10,6)
end = dt.datetime(2019,10,4)


# ticker_name = ''
# alpha = 0
# beta = 0
# average_returns = 0
# var = 0
# err = 0
# std_dev = 0
# r_sqrd = 0

# end = dt.datetime.now()
tickers = ['AAPL','MSFT']

# portfolio = Portfolio(tickers, start, end)

stocks = {}
rf_rate = 1.9 #Risk free rate. Set to 1.9%

portfolio = {}
excess_returns = {}

for ticker in tickers:
    temp = {}
    stocks[ticker] = Ticker(ticker, 'SPY', start, end)
    # temp['Stock'] = Ticker(ticker, 'SPY', start, end)
    temp['Excess Returns'] = stocks[ticker].average_return - (rf_rate/252)
    temp['Excess Returns over Beta'] = temp['Excess Returns'] / stocks[ticker].beta
    
    portfolio[ticker] = temp

    print('Processed %s'%ticker)
    print(portfolio[ticker])


print(stocks['AAPL'].plt_scatter())

col_3 = (portfolio['AAPL']['Excess Returns'] * stocks['AAPL'].beta) / stocks['AAPL'].err #unsystematic risk
print(col_3)

# col_4 = (stocks['AAPL'].beta ** 2) / stocks['AAPL'].err
# print(col_4)

# col_5 = 