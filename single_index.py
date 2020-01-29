
import pandas as pd
import datetime as dt
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from pandas_datareader import data
from ticker import *

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
# market_var

tickers = ['AMT','ARQL','XLY','VPU','GLPG','OLLI','MA','INFO','V','ECL','CHTR','AAPL']

stocks = {}
rf_rate = 1.9 #Risk free rate. Set to 1.9%

portfolio = {}
excess_returns = {}

sum_5 = 0
sum_6 = 0

portfolio = [] # Used for optimal portfolio

for ticker in tickers:
    temp = {}
    stocks[ticker] = Ticker(ticker, 'SPY', start, end)

    excess_returns = stocks[ticker].average_return - (rf_rate/252)

    temp['Ticker'] = ticker
    temp['Excess Returns']= excess_returns
    temp['Excess Returns over Beta'] = (excess_returns * stocks[ticker].beta)/stocks[ticker].err

    portfolio.append(temp)

    print('Processed %s'%ticker)

print(portfolio)
df = pd.DataFrame(portfolio)

## We want to sort tickers by excess returns over beta
df = df.sort_values(by=['Excess Returns over Beta'], ascending=False)

risk = df.iloc[0]['Excess Returns over Beta']

sum_col_3 = 0
sum_col_4 = 0
cutoff_point = 0 # Need to determine this
prev_cutoff = 0

sum_Z= 0

spy_var = stocks[tickers[0]].market_var
print("Spy Var = ", spy_var)

for index, row in df.iterrows():
    print(row['Ticker'])
    ticker = row['Ticker']
    stock = stocks[ticker]

    col_3 = excess_returns * stocks[ticker].beta/ stocks[ticker].err
    beta_risk = stocks[ticker].beta ** 2 / stocks[ticker].err

    sum_col_3 += col_3
    sum_col_4 += beta_risk

    df.at[index, 'Col 3'] =  col_3
    df.at[index, 'Col 4'] = beta_risk
    df.at[index, 'Sum Col 3'] = sum_col_3
    df.at[index, 'Sum Col 4'] = sum_col_4

    cutoff = spy_var * sum_col_3 / (1+spy_var * sum_col_4)

    df.at[index, 'Cutff'] = cutoff
    if (cutoff < prev_cutoff):
        cutoff_point = cutoff
    prev_cutoff = cutoff

print(cutoff_point)

for index, row in df.iterrows():
    ticker = row['Ticker']
    stock = stocks[ticker]

    print(stock.beta, stock.err, stock.average_return, rf_rate/252)

    Z = stock.beta / stock.err *  (((stock.average_return - rf_rate/252) / stock.beta) - cutoff_point)
    
    sum_Z += Z



    X = Z  / sum_Z
    pct_alloc = X * 100

    df.at[index, 'Z'] = Z
    df.at[index, 'X'] = X
    df.at[index, 'Pct Allocation'] = pct_alloc
  
print(df.head())

# col_5 = 