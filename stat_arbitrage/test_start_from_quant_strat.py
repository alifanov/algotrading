import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd
import statsmodels.api as sm

from sklearn import linear_model


def rolling_beta(X, y, idx, window=100):
    assert len(X) == len(y)

    out_dates = []
    out_beta = []

    model_ols = linear_model.LinearRegression()

    for iStart in range(0, len(X) - window):
        iEnd = iStart + window

        _x = X[iStart:iEnd].values.reshape(-1, 1)
        _y = y[iStart:iEnd].values.reshape(-1, 1)

        model_ols.fit(_x, _y)

        # store output
        out_dates.append(idx[iEnd])
        out_beta.append(model_ols.coef_[0][0])

    return pd.DataFrame({'beta': out_beta}, index=out_dates)


def create_pairs_dataframe(datadir, symbols):
    """Creates a pandas DataFrame containing the closing price
    of a pair of symbols based on CSV files containing a datetime
    stamp and OHLCV data."""

    # Open the individual CSV files and read into pandas DataFrames
    print("Importing CSV data...")
    sym1 = pd.read_csv(os.path.join(datadir, 'BTC_%s.csv' % symbols[0]),
                       header=0, index_col=0,
                       names=['date', 'open', 'high', 'low', 'close'])
    sym2 = pd.read_csv(os.path.join(datadir, 'BTC_%s.csv' % symbols[1]),
                       header=0, index_col=0,
                       names=['date', 'open', 'high', 'low', 'close'])

    # Create a pandas DataFrame with the close prices of each symbol
    # correctly aligned and dropping missing entries
    print("Constructing dual matrix for %s and %s..." % symbols)
    pairs = pd.DataFrame(index=sym1.index)
    pairs['%s_close' % symbols[0].lower()] = sym1['close']
    pairs['%s_close' % symbols[1].lower()] = sym2['close']
    pairs = pairs.dropna()
    return pairs


def calculate_spread_zscore(pairs, symbols, lookback=100):
    """Creates a hedge ratio between the two symbols by calculating
    a rolling linear regression with a defined lookback period. This
    is then used to create a z-score of the 'spread' between the two
    symbols based on a linear combination of the two."""

    # Use the pandas Ordinary Least Squares method to fit a rolling
    # linear regression between the two closing price time series
    s0 = symbols[0].lower()
    s1 = symbols[1].lower()

    print("Fitting the rolling Linear Regression...")

    ols = rolling_beta(pairs['%s_close' % s0],
                     pairs['%s_close' % s1],
                       pairs.index,
                     window=lookback)

    # Construct the hedge ratio and eliminate the first
    # lookback-length empty/NaN period
    pairs['hedge_ratio'] = ols['beta']
    pairs = pairs.dropna()

    # Create the spread and then a z-score of the spread
    print("Creating the spread/zscore columns...")
    pairs['hedge_ratio'] = [v for v in pairs['hedge_ratio'].values]
    pairs['spread'] = pairs['{}_close'.format(s0)] - pairs['hedge_ratio'] * pairs['{}_close'.format(s1)]
    pairs['zscore'] = (pairs['spread'] - np.mean(pairs['spread'])) / np.std(pairs['spread'])
    return pairs


def create_long_short_market_signals(pairs, symbols,
                                     z_entry_threshold=2.0,
                                     z_exit_threshold=1.0):
    """Create the entry/exit signals based on the exceeding of
    z_enter_threshold for entering a position and falling below
    z_exit_threshold for exiting a position."""

    # Calculate when to be long, short and when to exit
    pairs['longs'] = (pairs['zscore'] <= -z_entry_threshold) * 1.0
    pairs['shorts'] = (pairs['zscore'] >= z_entry_threshold) * 1.0
    pairs['exits'] = (np.abs(pairs['zscore']) <= z_exit_threshold) * 1.0

    # These signals are needed because we need to propagate a
    # position forward, i.e. we need to stay long if the zscore
    # threshold is less than z_entry_threshold by still greater
    # than z_exit_threshold, and vice versa for shorts.
    pairs['long_market'] = 0.0
    pairs['short_market'] = 0.0

    # These variables track whether to be long or short while
    # iterating through the bars
    long_market = 0
    short_market = 0

    # Calculates when to actually be "in" the market, i.e. to have a
    # long or short position, as well as when not to be.
    # Since this is using iterrows to loop over a dataframe, it will
    # be significantly less efficient than a vectorised operation,
    # i.e. slow!
    print("Calculating when to be in the market (long and short)...")
    for i, b in enumerate(pairs.iterrows()):
        # Calculate longs
        if b[1]['longs'] == 1.0:
            long_market = 1
        # Calculate shorts
        if b[1]['shorts'] == 1.0:
            short_market = 1
        # Calculate exists
        if b[1]['exits'] == 1.0:
            long_market = 0
            short_market = 0
        # This directly assigns a 1 or 0 to the long_market/short_market
        # columns, such that the strategy knows when to actually stay in!
        pairs.loc[pairs.index[i], 'long_market'] = long_market
        pairs.loc[pairs.index[i], 'short_market'] = short_market
    return pairs


def create_portfolio_returns(pairs, symbols):
    """Creates a portfolio pandas DataFrame which keeps track of
    the account equity and ultimately generates an equity curve.
    This can be used to generate drawdown and risk/reward ratios."""

    # Convenience variables for symbols
    sym1 = symbols[0].lower()
    sym2 = symbols[1].lower()

    # Construct the portfolio object with positions information
    # Note that minuses to keep track of shorts!
    print("Constructing a portfolio...")
    portfolio = pd.DataFrame(index=pairs.index)
    portfolio['positions'] = pairs['long_market'] - pairs['short_market']
    portfolio[sym1] = -1.0 * pairs['%s_close' % sym1] * portfolio['positions']
    portfolio[sym2] = pairs['%s_close' % sym2] * portfolio['positions']
    portfolio['total'] = portfolio[sym1] + portfolio[sym2]

    # Construct a percentage returns stream and eliminate all
    # of the NaN and -inf/+inf cells
    print("Constructing the equity curve...")
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['returns'].fillna(0.0, inplace=True)
    portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
    portfolio['returns'].replace(-1.0, 0.0, inplace=True)

    # Calculate the full equity curve
    portfolio['returns'] = (portfolio['returns'] + 1.0).cumprod()
    return portfolio


if __name__ == "__main__":
    datadir = 'datasets/'  # Change this to reflect your data path!
    symbols = ('ETC', 'LTC')

    # lookbacks = range(10, 310, 10)
    # returns = []

    lb = 20

    # Adjust lookback period from 50 to 200 in increments
    # of 10 in order to produce sensitivities
    # for lb in lookbacks:
    print("Calculating lookback=%s..." % lb)
    pairs = create_pairs_dataframe(datadir, symbols)
    pairs = calculate_spread_zscore(pairs, symbols, lookback=lb)
    pairs = create_long_short_market_signals(pairs, symbols,
                                             z_entry_threshold=2.0,
                                             z_exit_threshold=1.0)

    portfolio = create_portfolio_returns(pairs, symbols)
    portfolio['returns'].plot()
    plt.show()
    # returns.append(portfolio.iloc[-1]['returns'])
    print()

    # print("Plot the lookback-performance scatterchart...")
    # print('Best lookback: {}'.format(lookbacks[returns.index(max(returns))]))
    # plt.plot(lookbacks, returns, '-o')
    # plt.show()
#

# best lookback: 20