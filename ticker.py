import pandas as pd 
import datetime as dt 
import pandas_datareader.data as web
import pickle
from scipy import stats
import numpy as np 
import matplotlib.pyplot as plt

class Ticker:
    ticker_name = ''
    alpha = 0
    beta = 0
    average_returns = 0
    var = 0
    err = 0
    std_dev = 0
    r_sqrd = 0

    def plt_scatter(self):
        print(10*'-', self.ticker_name, ' Calcualted Variables', 10*'-')
        print('Alpha ', self.alpha)
        print('Beta ', self.beta)
        print('Average Return = ', self.average_return)    
        print('Var = ', self.var )
        print('Std Dev = ', self.std_dev)
        print('Error = ', self.err)
        print(45*'-')

        self.df.plot(x='Market Returns', y='Returns', kind='scatter')
        plt.show() 


    def calculate_regression_params(self):
        stock_returns = self.df['Returns']
        market_returns = self.df['Market Returns']
      
        slope, intercept, r_value, p_value, std_err = stats.linregress(market_returns,  stock_returns)

        self.r_sqrd = r_value ** 2
        self.beta = slope
        self.alpha = intercept
        self.average_return =  stock_returns.mean()

        variance = ( stock_returns- self.average_return)**2 / len( stock_returns) #Find variance in stock returns
        self.var = variance.sum()
        self.std_dev = np.sqrt(self.var)

        errors = (stock_returns - (self.alpha + self.beta *  market_returns)) ** 2
        self.err = errors.sum() / len(errors)
        # print('Error = ', total_error)

    def clean_dataframe(self):
        self.df['Returns'] = self.df['Adj Close'].pct_change() * 100
        self.df = self.df.dropna(subset=['Returns', 'Market Returns'])
        
        self.calculate_regression_params()


    """"
        Inputs:
            ticker - 
            corr_ticker - Ticker of stock to be correlated against
            start - Start date for data grabbing. Defaults to January 1st, 2016
            end - End data for data grabbing. Defaults to current date
    """
    def __init__(self, ticker, corr_ticker='SPY', start=dt.datetime(2016,1,1), end=dt.datetime.now()):
        self.ticker_name = ticker
        self.df = web.DataReader(ticker, 'yahoo', start, end)
        corr_df = web.DataReader(corr_ticker, 'yahoo', start, end)
        self.df['Market Returns'] = corr_df['Adj Close'].pct_change() * 100
        self.clean_dataframe()


# aapl = Ticker('AAPL', 'SPY')
# print(aapl.df.head())
# aapl.plt_scatter()
# print('DONE')