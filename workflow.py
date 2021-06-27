# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras


tt1 = pd.read_csv('<dataset file (data.csv) pathname>')

tt1.head()

tt1['timestamp'] =  pd.to_datetime(tt1['timestamp'])
tt1 = tt1.set_index('timestamp')
tt1.head()
tt = tt1
dataset = tt.values
dataset = dataset.astype('float32')
len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))
def create_training_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :11]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)
look_back = 5
trainX, trainY = create_training_dataset(train, look_back=look_back)
testX, testY = create_training_dataset(test, look_back=look_back)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.GRU(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(tf.keras.layers.Dense(11))
adamOpt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adamOpt, metrics=['mae'])
history = model.fit(trainX, trainY, validation_split=0.25,epochs=20, batch_size=64, verbose=2)


print("Predicting")
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainY = scaler.inverse_transform(trainY)
trainPredict = scaler.inverse_transform(trainPredict)
testY = scaler.inverse_transform(testY)
testPredict = scaler.inverse_transform(testPredict)

print("Evaluating Model")

trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:], testPredict[:]))
print('Test Score: %.2f RMSE' % (testScore))

print("Evaluation metrics: MAE(Mean Absolute Error)")

from sklearn.metrics import mean_absolute_error
trainScore = (mean_absolute_error(trainY[:], trainPredict[:]))
print('Train Score: %f MAE' % (trainScore))
testScore = math.sqrt(mean_absolute_error(testY[:], testPredict[:]))
print('Test Score: %f MAE' % (testScore))

trainScore2 = np.mean(np.abs(trainPredict - trainY)/np.abs(trainY))
print('Train Score: %f MAPE' % (trainScore2))
testScore2 = np.mean(np.abs(testPredict - testY)/np.abs(testY))
print('Test Score: %f MAPE' % (testScore2))

trainScore3 = np.corrcoef(trainPredict, trainY)[0,1]
print('Train Score: %f COR' % (trainScore3))
testScore2 = np.corrcoef(testPredict, testY)[0,1]
print('Test Score: %f COR' % (testScore2))

index=tt.index
TestY= pd.DataFrame(testY,columns=['min_cpu','max_cpu','avg_cpu','task_cpu','peakmemory','bandwidth','ioread','iowrite','core','memory','os'])
PredY=pd.DataFrame(testPredict,columns=['min_cpu','max_cpu','avg_cpu','task_cpu','peakmemory','bandwidth','ioread','iowrite','core','memory','os'])

x=index[-1722:]
fig, axs = plt.subplots(11,figsize=(10,15))

axs[0].plot(x,TestY.min_cpu,'.',label='Test min cpu',color='red')
axs[0].plot(x,PredY.min_cpu,'--',label='Predicted min cpu',color='black')
axs[0].legend()
axs[0].set(xlabel='Timestamp', ylabel='min cpu',autoscale_on=True)

axs[1].plot(x,TestY.max_cpu,'.',label='Test max cpu',color='magenta')
axs[1].plot(x,PredY.max_cpu,'--',label='Predicted max max cpu',color='navy')
axs[1].legend()
axs[1].set(xlabel='Timestamp', ylabel='max cpu',autoscale_on=True)

axs[2].plot(x,TestY.avg_cpu,'.',label='Test avg avg cpu',color='orange')
axs[2].plot(x,PredY.avg_cpu,'--',label='Predicted avg avg cpu',color='darkgreen')
axs[2].legend()
axs[2].set(xlabel='Timestamp', ylabel='avg cpu',autoscale_on=True)

axs[3].plot(x,TestY.task_cpu,'.',label='Test avg task_cpu',color='orange')
axs[3].plot(x,PredY.task_cpu,'--',label='Predicted avg task_cpu',color='darkgreen')
axs[3].legend()
axs[3].set(xlabel='Timestamp', ylabel='avg task_cpu',autoscale_on=True)

axs[4].plot(x,TestY.peakmemory,'.',label='Test avg peakmemory',color='orange')
axs[4].plot(x,PredY.peakmemory,'--',label='Predicted avg peakmemory',color='darkgreen')
axs[4].legend()
axs[4].set(xlabel='Timestamp', ylabel='avg peakmemory',autoscale_on=True)

axs[5].plot(x,TestY.bandwidth,'.',label='Test avg bandwidth',color='orange')
axs[5].plot(x,PredY.bandwidth,'--',label='Predicted avg bandwidth',color='darkgreen')
axs[5].legend()
axs[5].set(xlabel='Timestamp', ylabel='avg bandwidth',autoscale_on=True)

axs[6].plot(x,TestY.ioread,'.',label='Test avg I/O Read',color='orange')
axs[6].plot(x,PredY.ioread,'--',label='Predicted avg I/O Read',color='darkgreen')
axs[6].legend()
axs[6].set(xlabel='Timestamp', ylabel='avg I/O Read',autoscale_on=True)

axs[7].plot(x,TestY.iowrite,'.',label='Test avg I/O Write',color='orange')
axs[7].plot(x,PredY.iowrite,'--',label='Predicted avg I/O Write',color='darkgreen')
axs[7].legend()
axs[7].set(xlabel='Timestamp', ylabel='avg I/O write',autoscale_on=True)

axs[8].plot(x,TestY.core,'.',label='Test avg core',color='orange')
axs[8].plot(x,PredY.core,'--',label='Predicted avg core',color='darkgreen')
axs[8].legend()
axs[8].set(xlabel='Timestamp', ylabel='avg core',autoscale_on=True)

axs[9].plot(x,TestY.memory,'.',label='Test avg memory',color='orange')
axs[9].plot(x,PredY.memory,'--',label='Predicted avg memory',color='darkgreen')
axs[9].legend()
axs[9].set(xlabel='Timestamp', ylabel='avg memory',autoscale_on=True)

axs[10].plot(x,TestY.os,'.',label='Test avg OS',color='orange')
axs[10].plot(x,PredY.os,'--',label='Predicted avg OS',color='darkgreen')
axs[10].legend()

#for ax in axs.flat:
 #   ax.set(xlabel='Timestamp', ylabel='Workflow (CPU)',autoscale_on=True)
for ax in axs:
    ax.label_outer()


#fig.suptitle('Prediction of Workload on Azure cloud at a particular timestamp',fontsize=20)
plt.savefig('<provide output path for save>', dpi = 300)
plt.show()