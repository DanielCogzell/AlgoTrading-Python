# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:00:26 2019

General Template for Algo's

This Template:
    Generates the dataframe.
    Peforms some feature engineering
    Uses these new features and tries to make a prediction
    It then creates all the long and short positions
    Peforms a Mark to Market to calculate rolling PnL

The template still needs to add transaction costs and should account for
bid ask spreads.

@author: cogzelld
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

#symbols = ['USDZAR','EURZAR']

def featureEngineering(pairs,symbols):
    
    for sym in symbols:
        #CREATING NORMALISED MA
        for k in [3,5,10,15,20,30,60]:
            pairs["{}MA{}".format(sym,k)] = pairs[sym] - pairs[sym
                  ].rolling(k).mean()   #Will be negative if current is 
                                        #less than momentum
        for j in [1,5,10,15,20,30,60]:
            pairs["{}returns{}".format(sym,j)] = 100*pairs[sym].pct_change(j)
        
    pairs = pairs[60:] #DROPPING FIRST 60 ROWS CSE OF NANS
    pairs = pairs.dropna() #double sure
    return(pairs)
    
def predictOutcome(pairs,mins = 5):
    #CREATING DF WITH RESAMPLE FOR EVERY X MINS
    pairs = pairs.resample('{}T'.format(mins)).last()
    #COL TO PREDICT
    pairs['updown'] = pairs['USDZAR'].shift(-1) > pairs['USDZAR']
    #deleting first n and last n row cse of NaN
    pairs.drop(pairs.index[-mins:], inplace = True)
    
    pairs = pairs.dropna() #Making sure no Null values
    ##splitting into training and test data
    X_train = pairs[0:round(0.7*len(pairs))].drop('updown',1)
    y_train = pairs[0:round(0.7*len(pairs))]['updown']
    X_test = pairs[round(0.7*len(pairs)):].drop('updown',1)
    y_test = pairs[round(0.7*len(pairs)):]['updown']
    
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth = 2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    pairs = pairs[round(0.7*len(pairs)):]  
    pairs['pred'] = y_pred
    
    #Looking into confusion matrix
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    return(clf,pairs)

'''
Going long the spread when the z-score negatively exceeds a 
negative z-score threshold and short when it exceeds a positive
z-score threshold.
Exit signal is given when the absolute value of the z-score is 
less than or equal to another (smaller in magnitude) threshold.
(Might want to play around with entry and exit thresholds for max returns)
'''
def create_long_short_market_signals(pairs, pred):
    """Create the entry/exit signals based on the exceeding of 
    z_enter_threshold for entering a position and falling below
    z_exit_threshold for exiting a position."""

    # Calculate when to be long, short and when to exit
    pairs['longs'] = (pairs['pred'] > 0)*1.0    #make 0 true and false?
    pairs['shorts'] = (pairs['pred'] == 0)*1.0
    
    marketPos = []
    # Calculates when to actually be "in" the market, i.e. to have a
    # long or short position, as well as when not to be.
    # Since this is using iterrows to loop over a dataframe, it will
    # be significantly less efficient than a vectorised operation,
    # i.e. slow!
    print("Calculating when to be in the market (long and short)...")
    for i,j in enumerate(pairs.iterrows()):
        # These variables track whether to be long or short while
        # iterating through the bars
        # Calculate longs
        if pairs['longs'][i] == 1.0:
            marketPos.append(1)            
        # Calculate shorts
        elif pairs['shorts'][i] == 1.0:
            marketPos.append(-1)
            
    pairs['marketPos'] = marketPos
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

def create_portfolio_returns(pairs):
    """Creates a portfolio pandas DataFrame which keeps track of
    the account equity and ultimately generates an equity curve.
    This can be used to generate drawdown and risk/reward ratios."""

    # Construct the portfolio object with positions information
    # Note that minuses to keep track of shorts!
    print("Constructing a portfolio...")
    portfolio = pd.DataFrame(index=pairs.index)
    portfolio['positions'] = pairs['marketPos']

    #initialise returns
    portfolio['returns'] = 0
    returns = pairs['USDZAR'].pct_change().shift(-1)*pairs['position']+1
    
    #Position Difference
    portfolio['posChange'] = abs(portfolio['positions'][0:-1].
             values - portfolio['positions'][1:].values)
    
    print("Constructing the Returns curve...")
    #Returns IF TRADE, INCLUDE TRANSACTION COST
    for i in np.arange(1,len(portfolio)):
        if portfolio['posChange'][i] > 0: #posChange means there is a trade made
            if portfolio['posChange'][i]==1:
                returns[i] = returns[i]-tc 
            elif portfolio['posChange'][i]==2:
                returns[i] = returns[i]-2*tc 

    portfolio['returns'] = returns      
    portfolio['returns'].fillna(0.0, inplace=True)
    portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
    #Replace 100% loss with 0. happens if total goes from number to 0
    portfolio['returns'].replace(-1.0, 0.0, inplace=True)

    return portfolio

def actual(portfolio,betSize):
    portfolio['portfolio'] = portfolio['returns']*betSize
    
    return portfolio

if __name__ == "__main__":
    datadir = 'F:/Python Scripts/MyIdeas/CorrPairs/mostCorr.xlsx'  # Change this to reflect your data path!
    symbols = ['USDZAR', 'EURZAR']

    returns = [1]
    betSize = 5000000
    
    #CREATE DATAFRAME
    pairs = create_pairs_dataframe(datadir, symbols)
    '''
    #market conditions
    pairs = pairs.loc['2018-10-08':'2018-12-01'] #Bear1
    pairs = pairs.loc['2019-01-05':'2019-02-01'] #Bear2
    pairs = pairs.loc['2018-12-05':'2018-12-30'] #Bull1
    pairs = pairs.loc['2019-02-05':'2019-03-10'] #Bull2
    '''
    #PEFORM FEATURE ENGINEERING
    pairs = featureEngineering(pairs,symbols)
    
    #Transaction costs
    #Through Flex Trade there is only Bid Ask fees
    tc = 20/(pairs['USDZAR'].mean()*10000)
    
    #USE ML TO PREDICT OUTCOME (loop mins?)
    clf, pairs = predictOutcome(pairs,mins = 5)
    #CREATE LONG AND SHORT POSITIONS TO MEASURE MTM AT EACH POINT
    pairs = create_long_short_market_signals(pairs, pairs['pred'])
    #CALCULATE RETURNS OF MODEL
    portfolio = create_portfolio_returns(pairs)
    #STORE END RESULT. USED IF LOOPING THROUGH A METRIC. 
    returns.append(portfolio.iloc[-1]['returns'])
    #Determining actual portfolio
    portfolio = actual(portfolio,betSize)

    # This is still within the main function
    print("Plotting the performance charts...")
    fig = plt.figure(figsize=(12,9.3))
    fig.patch.set_facecolor('white')

    ax1 = fig.add_subplot(211,  ylabel='{} growth (%)'.format(symbols))
    plt.title('USDZAR')
    (pairs[symbols].pct_change()+1.0).cumprod().plot(ax=ax1, color='r', lw=2.)
    plt.grid(which='major')
    
    ax2 = fig.add_subplot(212, ylabel='Portfolio value growth (%)')
    portfolio['returns'].plot(ax=ax2, lw=2.)
    plt.grid(which='major')
    fig.show()