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
import matplotlib.pyplot as plt
from keras.callbacks  import EarlyStopping # checks if the updates are improving the model
from keras.optimizers import SGD # stochastic gradient descent optimiser 
from keras.layers     import Dense #type of connection between nodes (to everything)
from keras.models     import Sequential # type of model
from keras.utils      import to_categorical # allows us to create categorical variables
from keras.models     import load_model # allows us to load saved models

###############################################################################
# functions
def get_new_model(input_shape = input_shape):
 model = Sequential()
 model.add(Dense(100, activation='relu', input_shape = input_shape))
 model.add(Dense(100, activation='relu'))
 model.add(Dense(2, activation='softmax')) 
 return model

###############################################################################                 


###############################################################################
# Part 1: Defining a model
 
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

###############################################################################
#Part 2: Tuning a model

#1

# Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.01,1.0]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr = lr)
    
    # Compile the model
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target)

#2

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit the model
hist = model.fit(predictors, target, validation_split=0.3)


#3

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience = 2)

# Fit the model
model.fit(predictors, target, validation_split=0.3, epochs = 30, callbacks = [early_stopping_monitor])

#3 display validation score 
# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(100, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# MNIST

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'].)

# Fit the model
model.fit(X,y, validation_split = 0.3)

