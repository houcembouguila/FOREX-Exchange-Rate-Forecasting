#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd

import plotly as py
import os
import tensorflow as tf

import nbimporter
from feature_functions_2L_simple import *

from collections import deque

from alpha_vantage.timeseries import TimeSeries

import pickle

get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


from alpha_vantage.timeseries import TimeSeries


# In[24]:


cle_api = 'JBQR4M9YCQRT0EWV'


# In[25]:


ts= TimeSeries(key=cle_api, output_format='pandas')


# In[26]:


choice='svm'


# In[27]:


if choice=='svm':
    data0,meta_data=ts.get_intraday(symbol='EURUSD',interval='60min',outputsize='compact')
if choice == 'rnn':
    data0,meta_data=ts.get_intraday(symbol='EURUSD',interval='60min',outputsize='compact')
    #data0,meta_data=ts.get_daily(symbol='EURUSD',outputsize='compact')    


# In[28]:


# inverser la base
data = data0.loc[::-1,]


# In[29]:


#data.to_excel("Base_Courante.xlsx")


# In[30]:


Comission= 4/100000


# In[31]:


period_momentum = 10
period_bands = 15
period_RSI = 24
window_adx=14
cci_period=24


# In[32]:


data = data_prep(data,period_momentum,period_bands,period_RSI,window_adx,cci_period,comission=Comission)


# In[33]:


data.head()


# In[34]:


def choix_base(data,choix):
    
    if choix == 'svm':
        mydata=data[['momentum','RSI','PDI','NDI','ADX','CCI','return','label']]
        df=normalisation(mydata,choix)  

    if choix == 'rnn':
        mydata=data[['momentum','RSI','PDI','NDI','ADX','return','label']]
        df=normalisation(mydata,choix)
    
    return df


# In[35]:


df=choix_base(data,choice)
df.head()


# In[36]:


n_features=10


# In[37]:


def x_to_pred_func(df,choix):
    
    if choix=='svm':
        return (np.array(df.iloc[:,:-1]) )
    if choix=='rnn':
        return sequencement(df,n_features)


# In[38]:


x_to_predict = x_to_pred_func(df,choice)
x_to_predict.shape


# In[39]:


def prediction(x_to_predict,choix_modele):
    
    if choix_modele=='svm':
        
        with open('SVM_2L.pkl', 'rb') as f:
            model = pickle.load(f)
        
       # x_to_predict = np.reshape(x_to_predict,(x_to_predict.shape[0],x_to_predict.shape[1]*x_to_predict.shape[2]))
        predicted_y = model.predict(x_to_predict)
        
    if choix_modele=="rnn":
    
        model = tf.keras.models.load_model('RNN_2L')
        predicted_y = np.round(model.predict(x_to_predict))

    
    return(predicted_y)


# In[40]:


if choice == 'svm':
    taille_backtest = 0
if choice == 'rnn':
    taille_backtest = n_features-1
    


# In[41]:


from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(np.array(data['label'][taille_backtest:]), prediction(x_to_predict,choice))
print('\nTest Accuracy:{: .2f}%'.format(accuracy_test*100))


# In[42]:


# Predict test data

pred = prediction(x_to_predict,choice)


# Calculate equity..

contracts  = 10000.0
commission = 4/100000


df_trade = pd.DataFrame(np.array(data['return'][taille_backtest:]), columns=['return'])
df_trade['label']  = np.array(data['label'][taille_backtest:])
df_trade['pred']   = pred
df_trade['won']    = df_trade['label'] == df_trade['pred']
df_trade['return'] = df_trade['return'].shift(-1) 
#df_trade['return'] = df_trade['return'].shift(-1) * return_range
df_trade.drop(df_trade.index[len(df_trade)-1], inplace=True)

def calc_profit(row):
    if row['won']:
        return abs(row['return'])*contracts - commission*contracts
    else:
        return -abs(row['return'])*contracts - commission*contracts

df_trade['pnl'] = df_trade.apply(lambda row: calc_profit(row), axis=1)
df_trade['equity'] = df_trade['pnl'].cumsum()

display(df_trade.tail())

n_win_trades = float(df_trade[df_trade['pnl']>0.0]['pnl'].count())
n_los_trades = float(df_trade[df_trade['pnl']<0.0]['pnl'].count())
number_of_trades=float(df_trade['pnl'].count())
print("Profit Net         : $%.2f" % df_trade.tail(1)['equity'])
print("Nombre de prédictions justes : %d" % n_win_trades)
print("Nombre de prédictions fausses  : %d" % n_los_trades)
print("Nombre total de trades : %d" % number_of_trades)
print("Précision    : %.2f%%" % (100*n_win_trades/(n_win_trades + n_los_trades)))
print("Moyenne par Transaction Gagnée       : $%.3f" % df_trade[df_trade['pnl']>0.0]['pnl'].mean())
print("Moyenne par Transaction Perdue        : $%.3f" % df_trade[df_trade['pnl']<0.0]['pnl'].mean())
print("Gain le plus important    : $%.3f" % df_trade[df_trade['pnl']>0.0]['pnl'].max())
print("Perte la plus importante  : $%.3f" % df_trade[df_trade['pnl']<0.0]['pnl'].min())

df_trade['pnl'].hist(bins=20)

df_trade.plot(y='equity', figsize=(10,4), title='Test with $10000 initial capital')
plt.xlabel('Trades')
plt.ylabel('Equity (USD)')
for r in df_trade.iterrows():
    if r[1]['won']:
        plt.axvline(x=r[0], linewidth=0.5, alpha=0.8, color='g')
    else:
        plt.axvline(x=r[0], linewidth=0.5, alpha=0.8, color='r')


# In[ ]:




