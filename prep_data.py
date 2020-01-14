import pandas as pd 
import datetime as dt 
import pandas_datareader.data as web
import pickle

start = dt.datetime(2016, 6, 8)
end = dt.datetime.now()


def save_tickers(tickers, fn):
    with open(fn+'.pickle', 'wb') as f:
        pickle.dump(tickers, f)

def save_ticker_data(start=dt.datetime(2016,1,8),
                     end=dt.datetime.now(),
                     fn='output/portfolio_tickers'):

    with open(fn+'.pickle', 'rb') as f:
        tickers = pickle.load(f)

    prices_df = pd.DataFrame()
    prices_df = web.DataReader('SPY', 'yahoo', start, end)

    first_ticker = 1 #set first index to SPY
    for ticker in tickers:
        df = web.DataReader(ticker, 'yahoo', start, end)
        if (first_ticker):
            prices_df = pd.DataFrame(index=df.index)
            first_ticker = 0
        prices_df[ticker] = df['Adj Close']
    print(prices_df.head())



    return prices_df

# def get_tickers():
#     tickers = []
    
#     with open("portfolio_tickers.pickle", "wb") as f:
#         pickle.dump(tickers, f)
    
#     return tickers
