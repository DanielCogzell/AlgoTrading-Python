# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:16:50 2019

Dual Currency Strategy. Aim is to get a more steady return.

Conditions for a buy signal:
    RSI<70% ie not overbought or over sold.
    MACD > 0
    EMA: Price>20>50>100>200
    
@author: cogzelld
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##################### Functions #####################

def create_pairs_dataframe(datadir):
    pairs = pd.read_csv(datadir)
    #changing the date column to date time.
    pairs['Dates'] = pd.to_datetime(pairs['Unnamed: 0'])
    #setting the time column as the index
    pairs.set_index('Dates',inplace = True)
    pairs = pairs.dropna()
    return(pairs)
    
#Relative Strength Index  
def RSI(df, sym,n = 14):  
    UpMove = np.insert(np.array(df[sym][1:].values - df[sym][0:
        -1].values),0,0)
    DoMove = np.insert(np.array(df[sym][0:-1].values - df[sym][1:
        ].values),0,0)
    UpI = np.where(((UpMove[0:] > DoMove[0:]) & (UpMove[0:] > 0)),UpMove,0)
    DoI = np.where(((DoMove[0:] > UpMove[0:]) & (DoMove[0:] > 0)),DoMove,0)
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    UpIema = UpI.rolling(n).mean()
    DoIema = DoI.rolling(n).mean()
    RS = np.array(UpIema/DoIema)
    df['RSI{}'.format(sym)] = 100-(100/(1+RS))
    return df
    
def MACD(df, n_fast, n_slow):  
    EMAfast = pd.Series(df.ewm(span = n_fast, min_periods = n_slow - 1))  
    EMAslow = pd.Series(df.ewm(span = n_slow, min_periods = n_slow - 1))  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(df.ewm(MACD, span = 9, min_periods = 8), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df = df.join(MACD)  
    df = df.join(MACDsign)  
    df = df.join(MACDdiff)  
    return df

def featureEngineering(pairs,symbols):
    for sym in symbols:
        pairs['openDiff{}'.format(sym)] = pairs['openAsk{}'.format(sym)] - pairs['openBid{}'.format(sym)]
        pairs['highDiff{}'.format(sym)] = pairs['highAsk{}'.format(sym)] - pairs['highBid{}'.format(sym)]
        pairs['lowDiff{}'.format(sym)] = pairs['lowAsk{}'.format(sym)] - pairs['lowBid{}'.format(sym)]
        pairs['closeDiff{}'.format(sym)] = pairs['closeAsk{}'.format(sym)] - pairs['closeBid{}'.format(sym)]
        pairs['high{}'.format(sym)] = (pairs['highAsk{}'.format(sym)]+pairs['highBid{}'.format(sym)])/2
        pairs['low{}'.format(sym)] = (pairs['lowAsk{}'.format(sym)]+pairs['lowBid{}'.format(sym)])/2
        
        pairs = RSI(pairs,sym,14)
        
        for j in [12,20,26,50,100,200]:
            pairs['ema{}{}'.format(sym,j)] = pairs[sym].ewm(span = j).mean()
        
        pairs['MACD{}'.format(sym)] = (pairs['ema{}12'.format(sym)]-pairs['ema{}26'.format(sym)])
        
    pairs = pairs[200:]
        #pairs = pairs.dropna() #double sure
    return(pairs)
    
# x = percent for take profit y = percent for stop loss
def create_portfolio_returns(pairs, symbols, x = 3,y = 1): 
    #Calculate when to be long, short and when to exit
    portfolio = {}
    
    for sym in symbols:
        #initialise variables
        equity = [1/len(symbols)] #Even equity split to currency pairs
        pairs['pos'] = 0
        takeProf = 99999# must not be triggered in the first loop
        stopLoss = 0    # must not be triggered in the first loop
        shortTakeProf = 0       # must not be triggered in the first loop
        shortStopLoss = 99999   # must not be triggered in the first loop
        ret = pairs[sym].pct_change()
        
        #Enter when all conditions met
        #Longs
        pairs['longs{}'.format(sym)] = np.where(((pairs[sym]>pairs['ema{}20'.format(sym)]) & 
             (pairs['ema{}20'.format(sym)]>pairs['ema{}50'.format(sym)]) & 
             (pairs['ema{}50'.format(sym)]>pairs['ema{}100'.format(sym)]) &
             (pairs['ema{}100'.format(sym)]>pairs['ema{}200'.format(sym)]) & 
             (pairs['MACD{}'.format(sym)] > 0) & (pairs['RSI{}'.format(sym)] < 70)),1,0)

        pairs['longSig{}'.format(sym)] = np.insert(np.where(pairs['longs{}'.format(sym)][1:].values>
             pairs['longs{}'.format(sym)][0:-1].values,1,0),0,0)
        
        #Shorts
        pairs['shorts{}'.format(sym)] = np.where(((pairs[sym]<pairs['ema{}20'.format(sym)]) & 
             (pairs['ema{}20'.format(sym)]<pairs['ema{}50'.format(sym)]) & 
             (pairs['ema{}50'.format(sym)]<pairs['ema{}100'.format(sym)]) &
             (pairs['ema{}100'.format(sym)]<pairs['ema{}200'.format(sym)]) & 
             (pairs['MACD{}'.format(sym)] < 0) & 
             (pairs['RSI{}'.format(sym)] > 30)),1,0)
        
        pairs['shortSig{}'.format(sym)] = np.insert(np.where(pairs['shorts{}'.format(sym)][1:].values>
             pairs['shorts{}'.format(sym)][0:-1].values,1,0),0,0)

        for i in np.arange(0,len(pairs)):
            delta = ret[i]
            
            #Longs
            if (pairs['longSig{}'.format(sym)][i] == 1) & (pairs['pos'][i] == 0):
                #Buy signal
                pairs['pos'][i:] = 1
                #Exit at x% above or y% below entry
                takeProf = pairs['closeAsk{}'.format(sym)][i]*(1+x/100)
                stopLoss = pairs['closeAsk{}'.format(sym)][i]*(1-y/100)
                equity.append(equity[-1])
            elif (pairs['closeBid{}'.format(sym)][i] >= takeProf) & (pairs['pos'][i] == 1):
                pairs['pos'][i:] = 0
                equity.append(equity[-1]*(1+delta))
                #equity.append(equity[-1]*(1+x/100))
            elif (pairs['closeBid{}'.format(sym)][i] <= stopLoss) & (pairs['pos'][i] == 1):
                pairs['pos'][i:] = 0
                equity.append(equity[-1]*(1+delta))
                #equity.append(equity[-1]*(1-y/100))
            elif (pairs['pos'][i] == 1):
                equity.append(equity[-1]*(1+delta))
                
            #Shorts
            elif (pairs['shortSig{}'.format(sym)][i] == 1) & (pairs['pos'][i] == 0):
                #Buy signal
                pairs['pos'][i:] = -1
                #Exit at x% above or y% below entry
                shortTakeProf = pairs['closeBid{}'.format(sym)][i]*(1-x/100)
                shortStopLoss = pairs['closeBid{}'.format(sym)][i]*(1+y/100)
                equity.append(equity[-1])
            elif (pairs['closeAsk{}'.format(sym)][i] <= shortTakeProf) & (pairs['pos'][i] == -1):
                pairs['pos'][i:] = 0
                equity.append(equity[-1]*(1-delta))
                #equity.append(equity[-1]*(1+x/100))
            elif (pairs['closeAsk{}'.format(sym)][i] >= shortStopLoss) & (pairs['pos'][i] == -1):
                pairs['pos'][i:] = 0
                equity.append(equity[-1]*(1-delta))
                #equity.append(equity[-1]*(1-y/100))
            elif (pairs['pos'][i] == -1):
                equity.append(equity[-1]*(1-delta))
            else:
                pairs['pos'][i:] = pairs['pos'][i]
                equity.append(equity[-1]) #AMMEND NOT CONT!
                
        #individual returns for each pair
        portfolio['equity{}'.format(sym)] = equity
        portfolio['pos{}'.format(sym)] = pairs['pos']
    #sum of returns for each pair
    equity = np.array(portfolio['equityUSDZAR']) + np.array(portfolio['equityEURUSD'])
    return(pairs,equity,portfolio)

########  Portfolio  ############
symbols = ['USDZAR','EURUSD']   #Constituents
portfolio = {}                  #Initialise Portfolio 
               
for sym in symbols:
    datadir = 'Y:/Jhb/FOBO/Willem/General Data/{}_1m_Candle_Bid.csv'.format(sym)
    pairsBid = create_pairs_dataframe(datadir)
    pairsBid.columns = ['Unnamed: 0', 'openBid{}'.format(sym), 'highBid{}'.format(sym), 
                        'lowBid{}'.format(sym), 'closeBid{}'.format(sym)]
        
    datadir = 'Y:/Jhb/FOBO/Willem/General Data/{}_1m_Candle_Ask.csv'.format(sym)
    pairsAsk = create_pairs_dataframe(datadir)
    pairsAsk.columns = ['Unnamed: 0', 'openAsk{}'.format(sym), 'highAsk{}'.format(sym),
                        'lowAsk{}'.format(sym), 'closeAsk{}'.format(sym)]
    
    pairs = pd.concat([pairsBid,pairsAsk],axis = 1)
    pairs.drop(['Unnamed: 0'],1,inplace = True)
    
    pairs[sym] = (pairs['closeAsk{}'.format(sym)] + pairs['closeBid{}'.format(sym)])/2
    pairs = pairs[9:] #Rid of 9 mins of previous day.
    
    portfolio['{}'.format(sym)] = pairs.resample('1D').last()

######  Combining the currency dataframes ######
                           
pairs = pd.concat([portfolio[symbols[0]],portfolio[symbols[1]]],axis = 1)

###### Feature Engineering ######

pairs = featureEngineering(pairs,symbols)

maxReturns = [] 
minReturns = []
avgReturns = []
xx = [] #store of take profits looped through
yy = [] #store of stop losses looped through

#Loops through all possible combinations of take profits and stop losses through 
#the backtest period.
takeProfStart = 1
takeProfEnd = 3
stopLossStart = 1
stopLossEnd = 2
stepSize = 0.1

for x in np.arange(takeProfStart,takeProfEnd,stepSize): 
    for y in np.arange(stopLossStart,stopLossEnd,stepSize):
        pairs, equity, portfolio = create_portfolio_returns(pairs,symbols,x,y)
        maxReturns.append(max(equity))
        minReturns.append(min(equity))
        avgReturns.append(np.mean(equity))
        xx.append(x)
        yy.append(y)
    
        # This is still within the main function
        print("Plotting the performance charts...")
        fig = plt.figure(figsize=(12,9.3))
        fig.patch.set_facecolor('white')
    
        ax1 = fig.add_subplot(211,  ylabel='USDZAR')
        pairs['USDZAR'].plot(ax=ax1, color='r', lw=2.)
        plt.grid(which='major')
        
        ax2 = fig.add_subplot(212, ylabel='Portfolio value growth (%)')
        plt.plot(equity, lw=2.)
        plt.grid(which='major')
        fig.show()