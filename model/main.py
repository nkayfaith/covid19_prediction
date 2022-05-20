# -*- coding: utf-8 -*-
"""
Created on Fri May 20 08:58:34 2022

@author: nkayf
"""

#%% Imports and Paths

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from classes import ExploratoryDataAnalysis, ModelCreation,ModelTraining, ModelEvaluation

DATASET_TRAIN_PATH = (os.path.join(os.path.dirname(__file__), '..','data', 'cases_malaysia_train.csv'))
#DATASET_TEST_PATH = (os.path.join(os.path.dirname(__file__), '..','data', 'cases_malaysia_test.csv'))
LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_PATH = os.path.join(os.getcwd(),'model.h5')

#%% Step 1) Data Loading

X_train = pd.read_csv(DATASET_TRAIN_PATH)

#%% Step 2) Data Interpretation/Inspection

X_train.info()
X_train.isna().sum()
X_train.loc[~X_train['cases_new'].astype(str).str.isdigit(), 'cases_new'].tolist()

# =============================================================================
# Contains no NaN, but contains non-integers for 'new_cases'
# =============================================================================

#%% Step 3) Data Cleaning

eda = ExploratoryDataAnalysis()

# Change datatype 
X_train["cases_new"] = pd.to_numeric(X_train["cases_new"],errors="coerce")

# Impute NaN
X_train = X_train.loc[:,~X_train.columns.str.startswith('cluster')]
X_train = X_train.iloc[:,1:]
X_train = eda.impute_data(X_train)

# Set X_train
X_train = X_train.iloc[:,0]

# to visualise data distribution/trend
plt.figure()
plt.plot(X_train)
plt.show()

# =============================================================================
# Change 'new_cases' datatype to numeric
# Impute NaN
# Set X_train
# =============================================================================

#%% Step 4) Feature Selection
#%% Step 5) Data Preprocessing

#Encode
X_train_scaled = eda.scale_data(X_train)

window_size = 30 
data_size = X_train_scaled.shape[0]-30

X_train = []
y_train = []

#30 days
for i in range(window_size,data_size):
    X_train.append(X_train_scaled[i-window_size:i,0])
    y_train.append(X_train_scaled[i,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.expand_dims(X_train, axis=-1) 

# =============================================================================
# Scale all features using minmax because data contains no negative values
# =============================================================================

#%% Step 6) Model Building

mc = ModelCreation()
num_words = 10000
model = mc.lstm_layer(X_train,embedding_output=64, nodes=32, dropout=0.2)

# =============================================================================
# embedding_output=64, nodes=32, dropout=0.2, hidden_layer=2
# =============================================================================

#%% Step 7) Model Training

mt = ModelTraining()
hist = mt.model_training(model, X_train,y_train, epochs=100)
print(hist.history.keys())

# =============================================================================
# epochs = 100 with EarlyStopping
# =============================================================================

#%% Step 8) Model Performance
#%% Step 9) Model Evaluation

me = ModelEvaluation()
me.model_trend(hist)
me.model_evaluate(model,X_train,y_train)

# =============================================================================
#  Reviews
# * MAPE recorded at 0.0236%
# * Graph shows low loss, low mse which indicates model is good
# =============================================================================

#%% Step 10) Model Deployment
model.save(MODEL_PATH)

