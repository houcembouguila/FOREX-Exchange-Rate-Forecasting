#!/usr/bin/env python
# coding: utf-8

# In[2]:


#packages
import pandas as pd
#!pip install plotly
import plotly as py
from plotly import tools
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import mpl_finance
from mpl_finance import candlestick_ohlc
from matplotlib.dates import date2num
from datetime import datetime
import numpy as np

from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math

import import_ipynb
import nbimporter
from feature_functions_2L_simple import *

# Machine learning
import sklearn

from collections import deque

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score


# In[3]:


#DATA Online
#from alpha_vantage.timeseries import TimeSeries
#cle_api = 'JBQR4M9YCQRT0EWV'
#ts= TimeSeries(key=cle_api, output_format='pandas')
#data,meta_data=ts.get_intraday(symbol='EURUSD',interval='1min',outputsize='full')
#data.columns=['open','high','low','close','volume']
#print(data)


# In[26]:


#DATA
df=pd.read_csv('EURUSD_17_20.csv')
df.columns=['date','open','high','low','close','volume']
df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')
df=df.set_index(df.date)
df=df[['open','high','low','close','volume']]


# In[5]:


Comission = 4/100000


# In[6]:


#return_range = data_prep(df,comission=Comission)[1]


# In[7]:


data=data_prep(df,comission=Comission)


# In[8]:


from sklearn import preprocessing


# In[9]:


#data.dropna(inplace=True)


# In[10]:


#standardized_data = preprocessing.scale(data.iloc[:,:-2])


# In[11]:


mydata=data[['momentum','RSI','PDI','NDI','ADX','CCI','return','label']].copy()


# In[12]:


df=normalisation(mydata,'svm')
df.head()


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[14]:


X=np.array(df.iloc[:,:-1])
Y=np.array(df['label'])


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)


# In[20]:


import sklearn
from sklearn.svm import SVC


#svr_rbf = Pipeline([('rbf', SVC())])
svr_rbf=SVC(C=2.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                decision_function_shape='ovr', break_ties=False, random_state=None)
svr_rbf.fit(x_train, y_train)

accuracy_train = accuracy_score(y_train, svr_rbf.predict(x_train))
print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))

accuracy_test = accuracy_score(y_test, svr_rbf.predict(x_test))
print('\nTest Accuracy:{: .2f}%'.format(accuracy_test*100))


# Save model 

# In[22]:


import pickle
# now you can save it to a file
with open('SVM_2L.pkl', 'wb') as f:
    pickle.dump(svr_rbf, f)


# In[21]:


# Predict test data

pred = svr_rbf.predict(x_test)


# Calculate equity..

contracts  = 2000.0
commission = 4/100000


#df_trade = pd.DataFrame(train_x[train_len:,-1], columns=['return'])
df_trade = pd.DataFrame(np.array(data['return'][x_train.shape[0]:]), columns=['return'])
df_trade['label']  = y_test
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

df_trade.plot(y='equity', figsize=(10,4), title='Backtest with $10000 initial capital')
plt.xlabel('Trades')
plt.ylabel('Equity (USD)')
for r in df_trade.iterrows():
    if r[1]['won']:
        plt.axvline(x=r[0], linewidth=0.5, alpha=0.8, color='g')
    else:
        plt.axvline(x=r[0], linewidth=0.5, alpha=0.8, color='r')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


#indicators=np.array([df.momentum,df.RSI,df.PDI,df.NDI,df.ADX,df.CCI]).T


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


#matrice de confusion 
true_data = df_trade[df_trade['pred']==df_trade['label']]
false_data = df_trade[df_trade['pred']!=df_trade['label']]

#True positive
TP = true_data[true_data['label']==1].shape[0] 
print("TP = %d" % TP)
#True negative
TN = true_data[true_data['label']==0].shape[0] 
print("TN = %d" % TN)
#false positive 
FP = df_trade[df_trade['pred']==1].shape[0] - TP 
print("FP = %d" % FP)
#false negative
FN = df_trade[df_trade['pred']==0].shape[0] - TN 
print("FN = %d" % FN)

#True positive rate 
TPR = TP/(TP+FN)
print("TPR = %f" % TPR)

#False positive rate 
FPR = FP/(FP+TN)
print("FPR = %f" % FPR)


# In[20]:


#####################################################################################""


# In[21]:


#HA = heikenashi(df,[1])
#m = momentum(df,[10])
#res = m.close[10]


# In[ ]:





# In[ ]:




