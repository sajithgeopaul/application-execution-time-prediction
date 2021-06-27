# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from pandas_datareader import data


import io
df = pd.read_csv('<dataset file (data.csv) pathname>')
print(df,sep=',')
df.head()
print("With Normalization")
dataset = df.loc[:,'bandwidth'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'peakmemory'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'ioread'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'iowrite'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'task'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'memory'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'core'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'avg cpu'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'max cpu'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'min cpu'].values
dataset = np.reshape(dataset,(-1,1))
print(dataset)
dataset.shape



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print(dataset)
#train_size = int(len(dataset) * 0.80)
#test_size = len(dataset) - train_size
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

#split into samples


#split into samples
from numpy import array

#14600/365 = 40
#sample = list(rain)

#split a univariate sequence into samples
def split_sequence(sequence, n_steps):
  x, y = list(), list()
  for i in range(len(sequence)):
    #find end of this pattern
    end_ix = i + n_steps
    #check if beyond the sequence
    if end_ix > len(sequence)-1:
      break
    #gather input and output parts of the pattern
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
    x.append(seq_x)
    y.append(seq_y)
  return array(x), array(y)

series = array(dataset)
print(series.shape)

x, y = split_sequence(series, 365)
print(x.shape, y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#reshape 
series1 = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
series2 = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(series1.shape)
print(series2.shape)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping


K.clear_session()
model = Sequential()
model.add(Bidirectional(LSTM(13), input_shape=(365, 1)))
model.add(Dropout(0.1))
model.add(Dense(1))

#opt = keras.optimizers.Adam(learning_rate=1e-1000)
#opt = optimizers.adam(clipnorm=1.0)
#opt = SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
model.compile(loss='mean_squared_error', optimizer='adam')


history = model.fit(series1, y_train, epochs=100, batch_size= 200, validation_data=(series2, y_test), 
          callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

model.summary()

import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

train_predict = model.predict(series1)
test_predict = model.predict(series2)
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

print('Train Mean Absolute Error:', mean_absolute_error(y_train[:,0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(y_train[:,0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(y_test[:,0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:,0], test_predict[:,0])))

import matplotlib.pyplot as plt
#plot model loss

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();

#actual vs prediction

import seaborn as sns

aa=[x for x in range(100)]
plt.figure(figsize=(12,6))
plt.plot(aa, y_test[:,0][:100], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:100], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Avg annual accident rate Prediction', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();


#actual vs prediction

import seaborn as sns

aa=[x for x in range(200)]
plt.figure(figsize=(12,6))
plt.plot(aa, y_test[:,0][:200], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:200], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Execution time of the actual application ', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

print(test_predict.shape)
print(y_test.shape)

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predict)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 220]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


import scipy.stats as measures

per_coef = measures.pearsonr(y_test[:,0], test_predict[:,0])
#mse_coef = np.mean(np.square(np.array(y_pred) - np.array(y_true)))
print(per_coef)


per_coef1 = measures.pearsonr(y_train[:,0], train_predict[:,0])
print(per_coef1)


