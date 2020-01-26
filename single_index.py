
import pandas as pd
import datetime as dt
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from pandas_datareader import data



# stocks = ["SPY"]
# start = dt.datetime(2016,1,1)
# end = dt.datetime(2017,12,31)
start = dt.datetime(2016,10,6)
end = dt.datetime(2019,10,4)

# end = dt.datetime.now()
# stocks = ['AAPL','AMZN','GOOGL','FB', 'MSFT']
stocks = ['AAPL']

# data = quandl.get_table('WIKI/PRICES', ticker = stocks,
#                         qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
#                         date = { 'gte': '2016-1-1', 'lte': '2017-12-31' }, paginate=True)

# https://www.quandl.com/api/v3/databases/WIKI/metadata?api_key=%3CYOURAPIKEY%3E
# df = data.set_index('date')
# table = df.pivot(columns='ticker')
# # By specifying col[1] in below list comprehension
# # You can select the stock names under multi-level column
# table.columns = [col[1] for col in table.columns]

# returns = table.pct_change() # Daily pct change

## Can't get SPY from quandl for free
df = web.DataReader('AAPL', 'yahoo', start, end)
df_spy = web.DataReader('SPY', 'yahoo', start, end)

df_spy.reset_index(inplace=True)
# df_spy.drop(['High','Close','Low','Open','Volume'], axis=1,inplace=True)
df_spy.set_index("Date", inplace=True)

def calculate_CAPM(df_stock, df_market, show_plot=False):
    df_stock['Returns'] = df_stock['Adj Close'].pct_change() * 100
    df_market['Returns'] = df_market['Adj Close'].pct_change() * 100
    df_stock['Market Returns'] = df_market['Returns']
    df_subset = df_stock.dropna(subset=['Returns', 'Market Returns'])

    if show_plot:
        df_stock.plot(x='Market Returns', y='Stock Returns', kind='scatter')
        plt.show()

    stock_returns = df_subset['Returns']
    market_returns = df_subset['Market Returns']

    coeffs = np.polyfit( market_returns, stock_returns, 1)

    print(10*'-', 'Linear Regression Results', 10*'-')
    # TODO, these values are a factor off of the excel document
    beta = coeffs[1]
    alpha = coeffs[0]
    print('Alpha ', alpha)
    print('Beta ', beta)

    p = np.poly1d(coeffs)
    print('Equation is  ', p )

 
    slope, intercept, r_value, p_value, std_err = stats.linregress(market_returns,  stock_returns)
    print('Slope = ',slope)
    print('Intercept = ',intercept)
    print('R value = ',r_value)
    print('R squared = ' , r_value**2)
    print('P value = ',p_value)
    print('Std err = ',std_err)
    print(45*'-')

    beta = slope
    alpha = intercept
    average_return =  stock_returns.mean()
    print('Average Return = ', average_return)    

    variance = ( stock_returns- average_return)**2 / len( stock_returns) #Find variance in stock returns
    var = variance.sum()

    print('Var = ', var )
    print('Std Dev = ', np.sqrt(var))

    errors = (stock_returns - (alpha + beta *  market_returns)) ** 2
    total_error = errors.sum() / len(errors)
    print('Error = ', total_error)

calculate_CAPM(df, df_spy)
