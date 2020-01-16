# from data_prep import *
from portfolio_functions import *
from prep_data import *
from mean_variance import *
import sys, getopt

# start = dt.datetime(2016, 6, 8)
# end = dt.datetime.now()

# # ## Store ticker names in this pickle file
# with open("custom_stocks.pickle", "wb") as f:
#     pickle.dump(tickers, f)
# get_custom_stocks()

quandl.ApiConfig.api_key = ""



def main(argv):
    account_val = parse_arguments(argv) # Account value is passed from arguments
    if account_val <= 0:
        account_val = 20_000 # Default value is $20,000

    print("Account value is %0d" % account_val)
    risk_free_rate = 0.0152 # a 52 Week treasury bill at the start of 2018, was 1.52%

    stocks = ['AAPL','AMZN','GOOGL','FB', 'MSFT']
    data = quandl.get_table('WIKI/PRICES', ticker = stocks,
                            qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                            date = { 'gte': '2016-1-1', 'lte': '2017-12-31' }, paginate=True)

    df = data.set_index('date')
    table = df.pivot(columns='ticker')
    # By specifying col[1] in below list comprehension
    # You can select the stock names under multi-level column
    table.columns = [col[1] for col in table.columns]
    print(table.head())

    ## Mean-variance method
    display_ef_using_mean_variance(table, risk_free_rate)

    ### OLD WAY
    save_tickers(stocks, 'output/portfolio_tickers')
    prices_df = save_ticker_data()
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
            print('Account val is $%0d' % account_val)

    return int(account_val)

if __name__ == "__main__":
    main(sys.argv[1:])