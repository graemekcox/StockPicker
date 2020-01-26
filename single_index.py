
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


def calculate_beta(df_stock, market):
    stock_returns = df_stock.pct_change()

    variance = np.var(stock_returns)
    cov = stock_returns.cov(market)

    beta = cov/variance
    print(cov)
    print(beta)

    return beta

def calculate_CAPM(df_stock, df_market):
    df_stock['Returns'] = df_stock['Adj Close'].pct_change() * 100
    df_market['Returns'] = df_market['Adj Close'].pct_change() * 100


    # returns = df_stock.pct_change()
    # market_returns = df_market.pct_change()
    returns = df_stock['Returns']
    market_returns = df_market['Returns']

    average_return = returns.mean()
    df_stock['Average Return'] = average_return
    df_stock['Variance'] = ((df_stock['Returns']- average_return)**2) / len(df_stock['Returns'])

    # res = sm.ols(y = df_market['Returns'], x = df_stock['Returns'])
    # res = sm.ols(formula="Returns ~ 1 +")
    
    # df_stock['Returns']= df_stock['Returns'][df_stock['Returns'].replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)] 
    # df_market['Returns']= df_market['Returns'][df_market['Returns'].replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)] 

    # market_returns= market_returns[market_returns.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)] 


    df_stock['intercept'] = 1

    df_stock['Market Returns'] = df_market['Returns']
    df_subset = df_stock.dropna(subset=['Returns', 'Market Returns'])

    # df_stock.plot(x='Returns', y='Market Returns', kind='scatter')
    # plt.show()
    # print(df_subset.head())

    X = df_subset['Returns']
    y = df_subset['Market Returns']

    coeffs = np.polyfit( X, y, 1)

    print(10*'-', 'Linear Regression Results', 10*'-')
    # TODO, these values are a factor off of the excel document
    beta = coeffs[1]
    alpha = coeffs[0]
    print('Alpha ', alpha)
    print('Beta ', beta)

 
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    print('Slope = ',slope)
    print('Intercept = ',intercept)
    print('R value = ',r_value)
    print('R squared = ' , r_value**2)
    print('P value = ',p_value)
    print('Std err = ',std_err)
    print(45*'-')

    beta = slope
    alpha = intercept
    average_return = X.mean()
    print('Average Return = ', average_return)    
    
    
    # error = (df_subset['Returns'] - (alpha + beta* df_subset['Market Returns']))**2

    variance = (X- average_return)**2 / len(X) #Find variance in stock returns
    var = variance.sum()
    # var = var().mean()
    # print('Error = ', error)
    # print('Var = ', var)
    print('Var = ', var )
    print('Std Dev = ', np.sqrt(var))

    errors = (X - (alpha + beta * y)) ** 2
    total_error = errors.sum() / len(errors)
    print('Error = ', total_error)
    # std_dev
    #systematic_risk = sum( .error()) / 752

    # Systematic risk
    # unsystematic
    #Total risk = sd of stock returns /




    # fig, ax = plt.subplots()
    # ax.scatter(X,y, marker = '')

    # for i, label in enumerate(df_subset.index.values):
    #     ax.annotate(label, (X.iloc[i],y.iloc[i]))
    
    # ax.plot(np.unique(X),
    #     np.poly1d(np.polyfit(X,y,1))(np.unique(X)),
    #     color='black')

    # fig, ax = plt.subplots()
    # ax.scatter(X, y, marker='')

    # # for i, label in enumerate(df_subset.index.values):
    # #     ax.annotate(label, (X.iloc[i], y.i loc[i]))

    # # Fit a linear trend line
    # ax.plot(np.unique(X),
    #         np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
    #         color='black')

    
    # ax.set_xlim([3.3,10.5])
    # ax.set_ylim([4,10.5])
    # ax.set_xlabel('Market returns for AAPL')
    # ax.set_ylabel('Returns for SPY')
    # ax.set_title('OLS relationship between AAPL returns and SPY returns')

    # plt.show()
    # X = df_stock[['intercept','Returns']]
    # y = df_market['Returns']
    
    # res = sm.OLS( y, X).fit()
    # print(res)

    # df_stock.to_csv('aapl.csv')
    # df_market.to_csv('spy.csv')

    # returns= returns[returns.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)] 
    # market_returns= market_returns[market_returns.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)] 

    # df['Error'] = df['Return']- ()

    # unsystematic_risk = df['Error'].sum()/len(df.index)

    # model = sm.OLS( returns, market_returns).fit()
    #stock returns is dependent
    # Market returns is independent variable

    # model = sm.OLS( df_stock['Returns'],  df_market['Returns'], missing='drop').fit()

    # print("MODE PARAMS")
    # print(model.params)
    # print("SUmMARY")
    # print(model.summary())
    # print("Standard Errors")
    # print(model.bse)
    # print("Predicted values")
    # print(model.predict())

    # For linear regression, X = SPY Adj Close
    # Y = stock Adj close

    ### Unsystematic risk =
    # 
    #
    # print("ALPHA")

    # print("BETA") 



# calculate_beta(df['Adj Close'], df_spy['Adj Close'])

# df.to_csv('aapl.csv')
# df_spy.to_csv('spy.csv')
calculate_CAPM(df, df_spy)
