
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

tickers = ['AAPL','MSFT']

stocks = {}
rf_rate = 1.9 #Risk free rate. Set to 1.9%

portfolio = {}
excess_returns = {}

sum_5 = 0
sum_6 = 0


df = pd.DataFrame(index=tickers,columns=["Excess Returns",'Excess Returns with Beta'])

for ticker in tickers:
    temp = {}
    stocks[ticker] = Ticker(ticker, 'SPY', start, end)
    # stocks[ticker].plt_scatter()

    excess_returns = stocks[ticker].average_return - (rf_rate/252)

    # df['Excess Returns'][ticker] = excess_returns
    # df['Excess Returns over Beta'][ticker] = (excess_returns * stocks[ticker].beta)/stocks[ticker].err
    beta_risk = stocks[ticker].beta ** 2 / stocks[ticker].err
    
    sum_excess_returns += (excess_returns * stocks[ticker].beta)/stocks[ticker].err
    sum_excess_returns

    sum_beta_risk+= stocks[ticker].beta ** 2 / stocks[ticker].err

    portfolio[ticker] = temp
    print('Processed %s'%ticker)
    # portfolio[ticker]['Excess Returns'] = stocks[ticker].average_return - (rf_rate/252)
    # portfolio[ticker]['Excess Returns over Beta'] = (stocks[ticker].average_return - (rf_rate/252))/stocks[ticker].beta


    # print(portfolio[ticker])

# print(stocks['AAPL'].plt_scatter())

# col_3 = (portfolio['AAPL']['Excess Returns'] * stocks['AAPL'].beta) / stocks['AAPL'].err #unsystematic risk
# print(col_3)

# col_4 = (stocks['AAPL'].beta ** 2) / stocks['AAPL'].err
# print("COL 4 = ",col_4)



print(df.head())


# col_5 = 