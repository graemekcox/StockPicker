import pyfolio 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import datetime as dt 
import os

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# yahoo_financials = YahooFinancials(risky_assets)
# data = yahoo_financials.get_historical_price_data('2019-01-01', '2019-09-30', 'daily')

# prices_df = pd.DataFrame({
#     asset: {x['formatted_date']: x['adjclose'] for x in data[asset]['prices']} for asset in risky_assets
# })
# prices_df.head()
# tickers = ['AAPL']
# tickers = ['SPY','AAPL','MSFT','FB']

# start = dt.datetime(2016, 6, 8)
# end = dt.datetime.now()

# prices_df = pd.DataFrame()
# prices_df = web.DataReader('SPY', 'yahoo', start, end)

# first_ticker = 1
# for ticker in tickers:
#     # ticker = ticker[:-1]

#     df = web.DataReader(ticker, 'yahoo', start, end)
#     # df.reset_index(inplace=True, drop=False)


#     if (first_ticker):
#         prices_df = pd.DataFrame(index=df.index)
#         first_ticker = 0

#     prices_df[ticker] = df['Adj Close']


# print(prices_df.head())

## save to csv

# Calculate expected returns
def get_weights(df):
    mu = expected_returns.mean_historical_return(df)
    # S = risk_models.sample_cov(prices_df)
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()

    # OPtimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    # ef = EfficientFrontier(mu, S, weight_bounds=(-1,1))## allows shorting
    weights = ef.max_sharpe()
    ef.portfolio_performance(verbose=True)

    cleaned_weights = ef.clean_weights()
    ef.save_weights_to_file("weights.txt")  # saves to file
    print(cleaned_weights)
    return cleaned_weights


def post_processing_weights(df, weights, account_val):
    ## Post-processing weights
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=account_val)
    allocation, leftover = da.lp_portfolio()
    print(allocation)

    return allocation