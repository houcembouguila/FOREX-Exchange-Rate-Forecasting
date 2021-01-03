#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import mplfinance
from matplotlib.dates import date2num
from datetime import datetime
import numpy as np

from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math

import os
import time
import nbimporter
from feature_functions_2L_simple import *

import sklearn
from collections import deque


import imblearn


# In[2]:


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[3]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from numpy import sqrt


# 
# ## Preparing Data

# In[4]:


df=pd.read_csv('EURUSD_13_20.csv')
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


# In[ ]:





# In[8]:


from sklearn import preprocessing
#standardized_data = preprocessing.scale(data.iloc[:,:-2])
#data_norm = pd.DataFrame(standardized_data)


# In[9]:


mydata=data[['momentum','RSI','PDI','NDI','ADX','return','label']]


# In[10]:


df=normalisation(mydata,'rnn')
df.head()


# In[11]:


data_norm=df.copy()


# In[12]:


n_features=6
sequences = []
prev_hours = deque(maxlen=n_features)

for observation in data_norm.values: 
    #prev_hours.append([observation[-2]])
    prev_hours.append([x for x in observation[:-1]])  
    if len(prev_hours) == n_features: 
        sequences.append([np.array(prev_hours), observation[-1]])
        
import random
random.shuffle(sequences)


X = np.array([exemple[0] for exemple in sequences])
Y = np.array([int(exemple[1]) for exemple in sequences])


train_len = int(len(sequences)*0.85)
valid_len = int(len(sequences)*0.95)

Y.shape = (Y.shape[0],1)

x_train=X[0:train_len]
y_train=Y[0:train_len]
x_valid = X[train_len:valid_len]
y_valid = Y[train_len:valid_len]
x_test = X[valid_len:]
y_test= Y[valid_len:]
y_train =np.array([int(x) for x in y_train ])
y_valid = np.array([int(x) for x in y_valid ])
y_test = [int(x) for x in y_test ]


# In[13]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout,BatchNormalization,Activation
from keras.regularizers import l2

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(LSTM(32,kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.01)
               ,input_shape=(x_train.shape[1:]), return_sequences=True))
#model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(LSTM(32,kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.01),input_shape=(x_train.shape[1:])))
#model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Activation('relu'))

#model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))


# In[14]:


opt = tf.keras.optimizers.Adam(lr=0.001)#1e-6

#opt2=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=False, name="SGD")

#rms_opt=tf.keras.optimizers.RMSprop(learning_rate=0.001,momentum=0.0,name="RMSprop")
# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

BATCH_SIZE = 30
EPOCHS= 30

history = model.fit(
    x_train, y_train,
    validation_data=(x_valid,y_valid),
    batch_size=BATCH_SIZE,
    epochs = EPOCHS)


# In[15]:


score = model.evaluate(x_train, np.array(y_train), verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate(x_test, np.array(y_test), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[16]:



pred = np.round(model.predict(x_test).reshape(-1))


# Calculate equity..

contracts  = 2000.0

commission = 4/100000


#df_trade = pd.DataFrame(train_x[train_len:,-1], columns=['return'])
df_trade = pd.DataFrame(np.array(data['return'][valid_len+n_features-1:]), columns=['return'])
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

n_win_trades = float(df_trade[df_trade['pnl']>=0.0]['pnl'].count())
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


# In[17]:


#standardized_data = preprocessing.scale(data.iloc[:,:-2])


# In[20]:


model.save('RNN_2L',save_format='h5')

