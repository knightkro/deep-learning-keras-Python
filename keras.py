# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:25:01 2017

@author: KnightG
"""

# keras tutorial from datacamp

# Import necessary modules
import pandas as pd
import numpy as np
import keras
from keras.layers import Dense #type of connection between nodes (to everything)
from keras.models import Sequential # type of model
from keras.utils import to_categorical # allows us to create categorical variables
from keras.models import load_model # allows us to load saved models

# Load the data into a pandas data frame
df = pd.read_csv('titanic_all_numeric.csv') 

# Isolate the predictors
predictors = df.drop(['survived'], axis=1).as_matrix()

# extract shape of the predictors
n_cols = predictors.shape[1]

# use Keras's to_categorical to create a categorical output 
#(one column for each possibility)
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))

# Add the output layer. 2 Possible outputs. 'softmax' will output a probability
model.add(Dense(2, activation='softmax')) 

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)

# save the model to file
model.save('titanic_model.h5')

#load model
load_model = load_model('titanic_model.h5')

#use to make predictions from "data_to_predict_with"
#predictions = load_model.predict(data_to_predict_with)

# extract predictions
#probability_true = predictions[:,1]