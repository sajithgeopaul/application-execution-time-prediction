# application-execution-time-prediction
The code uses machine learning methods to predict execution time prediction of workflow applications run in cloud environments.
Data is procured from Azure VM trace that can be generated using Azure Log Analytics Workspace.
workflow.py file is executed first in the multi-stage machine learning method and this step uses Gated Recurrent Unit or GRU machine learning method to relate between pre-run and run-time parameters.
The MachineLearning _BLSTM file is then executed to perfor learning between the parameters and the execution time using time stamp values.
app.py is run in Anaconda 3.8 environment to generate the result.
Emsemble file is then run to compute the parameters using machine learning ensemble methods.
