# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:40:38 2019

Mean Reversion Implementation

@author: cogzelld
"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import linear_model

"""
import data into dataframe. 
Make sure that the data is consistant with regards to the time stamps.
"""
def create_pairs_dataframe(datadir,symbols):
    pairs = pd.read_excel(datadir,'Sheet2')
    #changing the date column to date time.
    pairs['Dates'] = pd.to_datetime(pairs['Dates'])
    #setting the time column as the index
    pairs.set_index('Dates',inplace = True)
    pairs = pairs[symbols]
    pairs = pairs.dropna()
    return(pairs)

'''
Rolling linear Regression 
(use this to implement rolling LDA or logistic regression for first algo)

Calculating rolling beta coefficient for linear Regression.
Default lookback window of 100 bars.
(Ideally want to loop through lookback window to find optimal lookback period).
Once the rolling beta coefficient is calculated we add it to the pairs 
DataFrame and drop the empty rows
'''

def rolling_regression(y, x, window=60):
    # === Add a constant if needed ========================================
        X = x.to_frame()
        X['c'] = 1
    # === Loop... this can be improved ====================================
        estimate_data = []
        for i in range(window, x.index.size+1):
            X_slice = X.values[i-window:i,:] # always index in np as opposed to pandas, much faster
            y_slice = y.values[i-window:i]
            coeff = np.dot(np.dot(np.linalg.inv(np.dot(X_slice.T, X_slice)), X_slice.T), y_slice)
            estimate_data.append(coeff[0] * x.values[window-1] + coeff[1])
    # === Assemble ========================================================
        estimate = pd.Series(data=estimate_data, index=x.index[window-1:]) 
        return estimate             

#symbols = ['USDZAR','EURZAR']
def calculate_spread_zscore(pairs, symbols, lookback=100):
    """Creates a hedge ratio between the two symbols by calculating
    a rolling linear regression with a defined lookback period. This
    is then used to create a z-score of the 'spread' between the two
    symbols based on a linear combination of the two."""
    
    # Use the pandas Ordinary Least Squares method to fit a rolling
    # linear regression between the two closing price time series
    print("Fitting the rolling Linear Regression...")
    ###
    #This is where you can add the other pairs
    ###
    #model = pairs[symbols].rolling(lookback).apply(rolling_beta())
    pairs['hedge_ratio'] = rolling_regression(y=pairs[symbols[0]], 
         x=pairs[symbols[1]], window=lookback)

    # Construct the hedge ratio and eliminate the first 
    # lookback-length empty/NaN period
    pairs = pairs.dropna()

    # Create the spread and then a z-score of the spread
    ###
    #mean and stdDev must be calculated on historical data. use .rolling
    ###
    print("Creating the spread/zscore columns...")
    #Spread is actual price - esitmated true price#
    pairs['spread'] = pairs['USDZAR'] - pairs['hedge_ratio']
    pairs['zscore'] = (pairs['spread'] - pairs['spread'].rolling(lookback).
         mean())/pairs['spread'].rolling(lookback).std()
    return pairs

'''
Going long the spread when the z-score negatively exceeds a 
negative z-score threshold and short when it exceeds a positive
z-score threshold.
Exit signal is given when the absolute value of the z-score is 
less than or equal to another (smaller in magnitude) threshold.
(Might want to play around with entry and exit thresholds for max returns)
'''
def create_long_short_market_signals(pairs, symbols, 
                                     z_entry_threshold=2.0, 
                                     z_exit_threshold=1.0):
    """Create the entry/exit signals based on the exceeding of 
    z_enter_threshold for entering a position and falling below
    z_exit_threshold for exiting a position."""

    # Calculate when to be long, short and when to exit
    pairs['longs'] = (pairs['zscore'] <= -z_entry_threshold)*1.0
    pairs['shorts'] = (pairs['zscore'] >= z_entry_threshold)*1.0
    pairs['exits'] = (np.abs(pairs['zscore']) <= z_exit_threshold)*1.0

    # These signals are needed because we need to propagate a
    # position forward, i.e. we need to stay long if the zscore
    # threshold is less than z_entry_threshold but still greater
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
        pairs.iloc[i]['long_market'] = long_market
        pairs.iloc[i]['short_market'] = short_market
    return pairs

'''
We need to create a portfolio to keep track of the market value of 
the positions. The first task is to create a positions column that 
combines the long and short signals. This will contain a list of 
elements from (1,0,−1), with 1 representing a long/market position,
 0 representing no position (should be exited) and −1 representing 
 a short/market position. 
The sym1 and sym2 columns represent the market values of USDZAR and 
EURZAR positions at the close of each bar. we sum them to produce a 
total market value at the end of every bar.
'''
def create_portfolio_returns(pairs, symbols):
    """Creates a portfolio pandas DataFrame which keeps track of
    the account equity and ultimately generates an equity curve.
    This can be used to generate drawdown and risk/reward ratios."""
    
    # Convenience variables for symbols
    sym1 = symbols[0]
    sym2 = symbols[1]
    
    pairs = pairs[round(0.7*len(pairs)):]
    
    # Construct the portfolio object with positions information
    # Note that minuses to keep track of shorts!
    print("Constructing a portfolio...")
    portfolio = pd.DataFrame(index=pairs.index)
    portfolio['positions'] = pairs['long_market'] - pairs['short_market']
    portfolio[sym1] = -1.0 * pairs[sym1] * portfolio['positions'] 
    portfolio[sym2] = pairs[sym2] * portfolio['positions']
    portfolio['total'] = portfolio[sym1] + portfolio[sym2]

    # Construct a percentage returns stream and eliminate all 
    # of the NaN and -inf/+inf cells
    print("Constructing the Returns curve...")
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['returns'].fillna(0.0, inplace=True)
    portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
    portfolio['returns'].replace(-1.0, 0.0, inplace=True)

    # Calculate the full equity curve
    portfolio['returns'] = (portfolio['returns'] + 1.0).cumprod()
    return portfolio

''' 
to calculate the optimal lookback period, we will assign a range to look 
through and we will need a metric to measure. We will choose the final 
total returns as our metric. 
(We would actually want to assign weights to all of these metrics like return,
risk, max drawdown etc and then loop through different lookback periods and 
entry and exit positions to find the optimal strategy).
The final task is to use matplotlib to create a line chart of 
lookbacks vs returns:
'''

if __name__ == "__main__":
    datadir = 'F:/Python Scripts/MyIdeas/CorrPairs/3mostCorr.xlsx'  # Change this to reflect your data path!
    symbols = ['USDZAR', 'EURZAR']

    lookbacks = range(230, 240, 60) # 230 is best.
    returns = []

    # Adjust lookback period from 50 to 200 in increments
    # of 10 in order to produce sensitivities
    for lb in lookbacks: 
        print("Calculating lookback=%s..." % lb)
        pairs = create_pairs_dataframe(datadir, symbols)
        
        #Transaction costs
        #Through Flex Trade there is only Bid Ask fees
        tc = 20/(pairs['USDZAR'].mean()*10000)
        
        pairs = calculate_spread_zscore(pairs, symbols, lookback=lb)
        pairs = create_long_short_market_signals(pairs, symbols, 
                                     z_entry_threshold=2.0, 
                                     z_exit_threshold=1.0)
        pairs = pairs[round(0.55*len(pairs)):]
        portfolio = create_portfolio_returns(pairs, symbols)
        returns.append(portfolio.iloc[-1]['returns'])

    print("Plot the lookback-performance scatterchart...")
    plt.plot(lookbacks, returns, '-o')
    plt.show()

    # This is still within the main function
    print("Plotting the performance charts...")
    fig = plt.figure(figsize=(12,10))
    fig.patch.set_facecolor('white')

    ax1 = fig.add_subplot(211,  ylabel='%s growth (%%)' % symbols[0])
    plt.title('USDZAR')
    (pairs[symbols[0]].pct_change()+1.0).cumprod().plot(ax=ax1, color='r', lw=2.)
    plt.grid(which='major')
    
    ax2 = fig.add_subplot(212, ylabel='Portfolio value growth (%%)')
    portfolio['returns'].plot(ax=ax2, lw=2.)
    plt.grid(which='major')
    fig.show()

'''
Need to add other currency pairs to the linear regression model.
Also need to create some sort of optimisation with lookback period and entry
and exit levels etc.
'''