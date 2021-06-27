# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans  
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_classification, make_regression
import six
import pandas as pd
import numpy as np
import argparse
import json
import re
import os
import sys
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
from sklearn.metrics import silhouette_samples

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot





X = pd.read_csv('<dataset file (data1.csv) pathname>') 
               #usecols=[ 'cpu_requested', 'mem_requested'])
#X = X[X['cpu_requested'].notnull() & X['mem_requested'].notnull()]

print('Sampled data description resource request')
print(X.describe())

print('Sampled data description resource usage')






#distortions

distortions = []
for i in range(1, 11):
    est1 = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
    est1.fit(X)
    distortions.append(est1.inertia_)
plt.plot(range(1,11), distortions, marker = 'o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortions')
plt.show()

import pandas as pd
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
import math
import numpy
import matplotlib.pyplot as plt, mpld3
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
from random import randint, sample, seed
from os import listdir, chdir
from os import path 
import gzip

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style

import pandas as pd
import csv, math , datetime
import time
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import style

import numpy 
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy
import matplotlib.pyplot as plt, mpld3
import pandas
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd





series = pd.read_csv('<dataset file (data1.csv) pathanme>', usecols=['max_cpu'], engine='python', skipfooter=3)
X = series.values
look_back = 1
print(len(series),len(X))
size = int(len(X) * 0.83)
print(size)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
print(len(test))
for t in range(len(test)):
    model = ARIMA(history, order=(1,2,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('time =%f predicted=%f, expected=%f' % (t,yhat, obs))
error = mean_squared_error(test, predictions)

# calculate root mean squared error (test set)
testScore = math.sqrt(mean_squared_error(test, predictions))
print('ARIMA %.4f RMSE' % (testScore))

 #error computation
summation = 0

for i in range(len(test)):
    summation = summation + ((test[i]-predictions[i])/test[i])

n=len(test)
accuracy = 100-((1/n)*summation*100)

trainPredictPlot = numpy.empty_like(series)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(train)+look_back, :] = train
# shift test predictions for plotting
fig = plt.figure()
testPredictPlot = numpy.empty_like(series)
testPredictPlot[:, :] = numpy.nan



# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

dataframe = pd.read_csv('<dataset file (data1.csv) pathname>', usecols=['max_cpu'], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
avgVol= dataframe.mean()

# split into train and test sets
train_size = int(len(dataset) * 0.83)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(train_size,test_size)
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(1, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY,epochs=100, batch_size=1, verbose=2)
#make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
print(len(testY))
print(len(testPredict))


# invert predictions
"""trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])"""



 

# calculate root mean squared error


testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('LSTM %.4f RMSE' % (testScore))

"""newV= avgVol+trainScore
acc = 100-(((newV-avgVol)/(avgVol))*100)"""

# shift train predictions for plotting
"""trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict"""
# shift test predictions for plotting
"""testPredictPlot = numpy.empty_like(testPredict)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(testPredict)+(look_back*2)+1:len(testPredict)-1, :] = testPredict
print(avgVol)
print(newV)
print('accuracy % of',acc)"""

# plot baseline and predictions
"""plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.ylabel('CPU Usage')
plt.xlabel('Datapoints')
plt.show()"""




# fix random seed for reproducibility
numpy.random.seed(7)

dataframe = pd.read_csv('<dataset file (data1.csv) pathname>', usecols=['max_cpu'], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
avgv=dataframe.mean()
# split into train and test sets
train_size = int(len(dataset) * 0.83)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
def create_dataset(dataset, look_back=10):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=2, verbose=2)
# Estimate model performance
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.4f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('MLP %.4f MSE (%.24f RMSE)' % (testScore, math.sqrt(testScore)))

newV= avgv+math.sqrt(trainScore)
acc = 100-(((newV-avgv)/(avgv))*100)
#print(avgv)
print('Mean Absolute Error MLP =' '',newV)
print('Accuracy=''%',acc)

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict1 = model.predict(testX)





df = pd.read_csv('<dataset file (data1.csv) pathname>')

forcast_col='max_cpu'
#the column that we want to predict
df.fillna(-99999, inplace=True)
#print(df)
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forcast_col].shift(-forecast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
#This contains all the data till the forecasy_out value
X = X[:-forecast_out:]
df.dropna(inplace=True)
#print(df)
df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])
#print(len(X),len(y))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.17)
#the size of the testing data
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(X_train)
model = svm.SVR(kernel='linear',C=1e3, gamma=0.0002)

model.fit(X_train,y_train)
accuracy = model.score(X_test,y_test)
predicted = model.predict(X_test)

"""
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last
one_day = 10368000
next_unix = last_unix + one_day

for i in forecast_set:
next_date = datetime.datetime.fromtimestamp(next_unix)
next_unix += one_day
df.loc[next_date]= [np.nan for _ in range(len(df.columns)-1)] + [i]
"""
"""df['cpu_usage'].plot()
df['label'].plot()
plt.legend(loc=4)
plt.xlabel('Time Steps')
plt.ylabel('CPU Usage')
plt.title('SVM')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(scaler.inverse_transform(df['cpu_usage']),label='Actual', color = 'blue')
ax.plot(predicted, label='Predicted', color = 'green')
ax.plot(X_test, label='Test', color = 'red')
#plt.xlim(min(dataset['time']), max(dataset['time']))"""

"""# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
print(avgVol)
print(newV)
print('accuracy % of',acc)"""

# plot baseline and predictions
"""plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.ylabel('CPU Usage')
plt.xlabel('Datapoints')
plt.show()"""

"""fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(scaler.inverse_transform(dataset),label='Actual', color = 'blue')
ax.plot( trainPredictPlot, label='Training', color = 'red')
ax.plot(testPredictPlot, label='Predicted', color = 'green')
#plt.xlim(min(dataset['time']), max(dataset['time']))

plt.xlabel('Time')
plt.ylabel('Memory Usage')
plt.title('LSTM')
plt.legend()
plt.show()
"""


"""predictedPlot[:, :] = numpy.nan
predictedPlot[len(predicted)+(look_back*2)+1:len(predicted)-1, :] = predicted"""
#predicted = numpy.reshape(predicted,1)
print (predictions)
print(predicted)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(test, color = 'blue', label='Actual')
plt.plot(predictions, label = 'ARIMA')
plt.plot(testPredict1, label='MLP')
plt.plot(testPredict, label='LSTM')
#plt.plot(predictedPlot, label='SVM')


#plt.title('ARIMA')
plt.ylabel('CPU Load ')
plt.xlabel('Steps')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(test, color = 'blue', label='Actual')
plt.plot(predictions, label = 'ARIMA')
plt.plot(testPredict1, label='MLP')
plt.plot(testPredict, label='LSTM')


#plt.title('ARIMA')
plt.ylabel('Memory usage ')
plt.xlabel('Steps')
plt.legend()
plt.show()

print (predicted)

print (np.transpose(predicted))
