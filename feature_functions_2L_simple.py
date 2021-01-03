#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#!pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
import mpl_finance
from mpl_finance import candlestick_ohlc
from matplotlib.dates import date2num
from datetime import datetime



# ## Heiken Ashi

# In[1]:


def HA(df):
    df['HAclose']=(df['open']+ df['high']+ df['low']+df['close'])/4

    idx = df.index.name
    df.reset_index(inplace=True)

    for i in range(0, len(df)):
        if i == 0:
            df.set_value(i, 'HAopen', ((df.get_value(i, 'open') + df.get_value(i, 'close')) / 2))
        else:
            df.set_value(i, 'HAopen', ((df.get_value(i - 1, 'HAopen') + df.get_value(i - 1, 'HAclose')) / 2))

    if idx:
        df.set_index(idx, inplace=True)

    df['HAhigh']=df[['HAopen','HAclose','high']].max(axis=1)
    df['HAlow']=df[['HAopen','HAclose','low']].min(axis=1)
    


# # Momentum

# In[1]:


#Momentum function 
def momentum(prices,periods):
    
    results= pd.DataFrame()
    open={}
    close={}
    
    for i in range(0,len(periods)):
        open[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:] - prices.open.iloc[:-periods[i]].values,
                                       index= prices.open.iloc[periods[i]:].index )
        close[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values,
                                       index= prices.close.iloc[periods[i]:].index )
        open[periods[i]].columns = ['open']
        close[periods[i]].columns = ['close']
    
    results.open = open
    results.close = close
    
    return results


# # Bollinger Bands 

# In[ ]:


def Bolinger_Bands(stock_price, window_size, num_of_std):

    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)

    return rolling_mean, upper_band, lower_band


# ## RSI

# In[2]:


def RSI(price,window_length):
    up, down = price.copy(), price.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    # Calculate the SMA
    roll_up2 = up.rolling(window_length).mean()
    roll_down2 = down.abs().rolling(window_length).mean()

    # Calculate the RSI based on SMA
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))
    
    return RSI2


# ## ADX
# 

# require from ta.trend import ADXIndicator 

# In[ ]:


from ta.trend import ADXIndicator


# In[3]:


def ADX(df,window_adx=14):
    
    adxI = ADXIndicator(df['high'],df['low'],df['close'],window_adx,False)
    df['PDI'] = adxI.adx_pos()
    df['NDI'] = adxI.adx_neg()
    #df['adx'] = adxI.adx()
    df['ADX'] =  ( ( adxI.adx_pos()-adxI.adx_neg() )/(adxI.adx_pos()+adxI.adx_neg()) ).rolling(14).mean()
    return(df)


# ## CCI

# In[ ]:


def CCI(data, ndays): 
    TP = (data['high'] + data['low'] + data['close']) / 3 
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),
                    name = 'CCI') 
    data = data.join(CCI) 
    return data


# ## Data Prep
# 

# creation label

# In[2]:


def creation_label(variation,comission):
    if variation > 4*comission:
        return 1
    elif variation< - 4*comission:
        return 0
 


# data

# In[1]:


def data_prep(data,period_momentum=10,period_bands=15,period_RSI=24,window_adx=14,cci_period=24,comission=4/100000):
    Data = data
    #Data =  Data.loc[::-1,]

    Data.columns = ['open','high','low','close','volume']

    Data.drop(columns='volume',inplace=True)
    
    #return and label
    Data['return'] = Data['close'] - Data['close'].shift(1)
    #return_range = Data['return'].max() - Data['return'].min()
    
    Data['label'] = Data['return'].shift(-1).apply(lambda x: 1 if x>0.0 else 0)
    
    #Data['return'] = Data['return'] / return_range
    
    #CCI
    Data = CCI(Data,cci_period)
  
    
    #ADX
    ADX(Data,window_adx)
    
    
    #momentum
    res = momentum(Data,[period_momentum]).close[period_momentum]
    #Base=Base.loc[Base.index[10]:]
    Data['momentum'] = res
    
    
    #heiken Ashi
    HA(Data)
    
    
    #Bollinger bands
    MIDband,UPband,LOWband = Bolinger_Bands(Data.close,period_bands,2)
    data_Bands = pd.DataFrame({'MIDband': MIDband, 'UPband': UPband,'LOWband':LOWband})
    df=pd.concat([Data, data_Bands], axis=1)
    
    #RSI
    df['RSI'] = RSI(df['return'],period_RSI)
    

    
    #final data
    DF = df.loc[:,['HAclose','momentum','MIDband','UPband','LOWband','RSI','PDI','NDI','ADX','CCI',
                   'return','label']]
    DF.dropna(inplace=True)
    
    
    return(DF)


# In[ ]:


def normalisation(data,choix):
    
    momentum_ecartype =data['momentum'].std()
    momentum_mean =data['momentum'].mean()
    
    return_ecartype =data['return'].std()
    return_mean = data['return'].mean()
    
    RSI_ecartype =data['RSI'].std()
    RSI_mean = data['RSI'].mean()
    
    ADX_ecartype =data['ADX'].std()
    ADX_mean = data['ADX'].mean()
    
    PDI_ecartype =data['PDI'].std()
    PDI_mean = data['PDI'].mean()
    
    NDI_ecartype =data['NDI'].std()
    NDI_mean = data['NDI'].mean()
    

   
    data.loc[:,'momentum'] = data['momentum'] - momentum_mean
    data.loc[:,'momentum'] = data['momentum']/momentum_ecartype
    
    data.loc[:,'return'] = data['return'] - return_mean
    data.loc[:,'return'] = data['return']/return_ecartype  
    
    data.loc[:,'RSI'] = data['RSI'] - RSI_mean
    data.loc[:,'RSI'] = data['RSI']/RSI_ecartype
    
    data.loc[:,'ADX'] = data['ADX'] - ADX_mean
    data.loc[:,'ADX'] = data['ADX']/ADX_ecartype
    
    data.loc[:,'PDI'] = data['PDI'] - PDI_mean
    data.loc[:,'PDI'] = data['PDI']/PDI_ecartype
    
    data.loc[:,'NDI'] = data['NDI'] - NDI_mean
    data.loc[:,'NDI'] = data['NDI']/NDI_ecartype
    
    if(choix=='svm'):   
        CCI_ecartype =data['CCI'].std()
        CCI_mean = data['CCI'].mean()
        data.loc[:,'CCI'] = data['CCI'] - CCI_mean
        data.loc[:,'CCI'] = data['CCI']/CCI_ecartype
    
    return(data)


# In[ ]:


from collections import deque


# In[2]:


def sequencement(data,n_features=10):
    
    sequences = []
    prev_hours = deque(maxlen=n_features)

    for observation in data.values: 
        #prev_hours.append([observation[-2]])
        prev_hours.append([x for x in observation[:-1]])  
        if len(prev_hours) == n_features: 
            sequences.append([np.array(prev_hours), observation[-1]])
        

    X = np.array([exemple[0] for exemple in sequences])
    Y = np.array([int(exemple[1]) for exemple in sequences])

    Y.shape = (Y.shape[0],1)

    #return (X,Y)
    return (X)

