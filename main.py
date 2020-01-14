# from data_prep import *
from portfolio_functions import *

import sys, getopt

# start = dt.datetime(2016, 6, 8)
# end = dt.datetime.now()

# # ## Store ticker names in this pickle file
# with open("custom_stocks.pickle", "wb") as f:
#     pickle.dump(tickers, f)
# get_custom_stocks()

def main(argv):
    account_val = parse_arguments(argv) # Account value is passed from arguments
    if account_val <= 0:
        account_val = 20_000 # Default value is $20,000

    print("Account value is ",account_val)
    
    tickers = ['SPY','AAPL','MSFT','FB']

    # account_val = 20000 #how much money in your account


    start = dt.datetime(2016, 6, 8)
    end = dt.datetime.now()

    prices_df = pd.DataFrame()
    prices_df = web.DataReader('SPY', 'yahoo', start, end)

    first_ticker = 1
    for ticker in tickers:

        df = web.DataReader(ticker, 'yahoo', start, end)
        # df.reset_index(inplace=True, drop=False)

        if (first_ticker):
            prices_df = pd.DataFrame(index=df.index)
            first_ticker = 0

        
        prices_df[ticker] = df['Adj Close']


    print(prices_df.head())


    # compile_data()

    # df = pd.read_csv('stock_tickers/AAPL.csv')

    # visualize_data()

    weights = get_weights(prices_df)
    post_processing_weights(prices_df, weights, account_val)

def parse_arguments(argv):
    account_val = 0
    try:
        opts, args = getopt.getopt(argv,"ha:",["account_val="])
    except getopt.GetoptError:
        print('test.py -a <account_value>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -a <account_value>')
            sys.exit()
        elif opt in ("-a", "--account_val"):
            account_val = arg

    print('Account val is $', account_val)

    return int(account_val)

if __name__ == "__main__":
    print("START TEST")
    main(sys.argv[1:])