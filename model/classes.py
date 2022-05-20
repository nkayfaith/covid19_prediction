# -*- coding: utf-8 -*-
"""
Created on Fri May 20 08:57:54 2022

@author: nkayf
"""
#%% Imports and Paths

import os
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_absolute_error

MODEL_PATH = os.path.join(os.getcwd(),'model.h5')
LOG_PATH = os.path.join(os.getcwd(),'log')

#%% Classes and Function

class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def label_encode(self, data):
        # Convert to numeric
        le = LabelEncoder()
        df_temp = data.astype("object").apply(le.fit_transform)
        data = df_temp.where(~data.isna(), data)
        return data
    
    def one_hot_encoder(self,data):
        '''
        This function will encode input data using one hot encoder approach
    
        Parameters
        ----------
        input_data : List,Array
            Input Data will undergo one-hot encoding.
    
        Returns
        -------
        encoded(input_data) : Array
            Input Data will undergo one-hot encoding.
    
        '''
        enc = OneHotEncoder(sparse=False)
        return enc.fit_transform(np.expand_dims(data,axis=-1))
    
    def impute_data(self,data):
        imputer = KNNImputer(n_neighbors=5)
        data = imputer.fit_transform(data) 
        data = pd.DataFrame(data)
        return data
    
    def feature_selection(self,data):         
        plt.figure()
        sns.heatmap(data.corr(), annot=True, cmap=plt.cm.Reds)
        plt.show()
        
    def scale_data(self,data):
        mms_scaler = MinMaxScaler()
        return mms_scaler.fit_transform(np.expand_dims(data,-1))
   
class ModelCreation():
    def lstm_layer(self,data,embedding_output=128, nodes=64, dropout=0.2):
        model = Sequential()
        model.add(LSTM(128, activation='tanh', 
                       return_sequences=(True), 
                       input_shape=(data.shape[1:])))        
        model.add(Dropout(dropout))
        model.add(LSTM(nodes))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer='adam',
                          loss='mse',
                          metrics='mse')
        model.summary()
        
        return model
    
    def model_plot(self,model):
        plot_model(model)
        
class ModelTraining():
    def model_training(self,model, x_train,y_train, validation_data=None,epochs=100):
        log_files = os.path.join(LOG_PATH,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
        early_stopping_callback = EarlyStopping(monitor='loss',patience=3)
        
        return model.fit(x_train,y_train, epochs=epochs, validation_data=validation_data,callbacks=[tensorboard_callback,early_stopping_callback])

class ModelEvaluation():
    def model_trend(self,hist):
       plt.figure()
       plt.plot(hist.history['loss'])    
       plt.show()
        
       plt.figure()
       plt.plot(hist.history['mse'])    
       plt.show()

    def model_evaluate(self,model,X,y):
        predicted = []
        
        for i in X:
            predicted.append(model.predict(np.expand_dims(i, axis=0)))
            
        predicted = np.array(predicted)
        
        plt.figure()
        plt.plot(y)    
        plt.plot(predicted.reshape(len(predicted),1))    
        plt.legend(['actual','predicted'])
        plt.show() 
              
        y_true = y
        y_pred = predicted.reshape(len(predicted),1)
        
        print("Mean Absolute Percentage Error\t: {:.4f} %".format((mean_absolute_error(y_true, y_pred)/sum(abs(y_true)))*100))

  