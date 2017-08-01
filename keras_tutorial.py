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
def get_sequential_model(input_nodes, first_layer, hidden_layer, output_shape, activation_input, activation_hidden, activation_output):
 """ Define a dense sequential neural network with input_nodes number of input nodes,
 hidden_layer a list of hidden layers, output_shape defines output shape. In addition
 the types of activation need to be defined. """
 model = Sequential() # set up model
 model.add(Dense(first_layer, activation = activation_input, input_shape = input_nodes)) # first layer
 for layer in hidden_layer: # set up further hidden layers
     model.add(Dense(layer, activation = activation_hidden))
 model.add(Dense(output_shape, activation = activation_output)) 
 return model

###############################################################################                 


###############################################################################
# Get some data
 
# Load the data into a pandas data frame
df = pd.read_csv('C:\\Users\\KnightG\\Dropbox\\repos\\deep-learning-keras-Python\\titanic_all_numeric.csv') 

# Isolate the predictors
predictors = df.drop(['survived'], axis=1).as_matrix()

# extract shape of the predictors
n_cols = predictors.shape[1]

# use Keras's to_categorical to create a categorical output 
#(one column for each possibility)
target = to_categorical(df.survived)



###############################################################################
# Set up a model
model = get_sequential_model((n_cols,), 32, [], 2, 'relu', 'relu', 'softmax')

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
#Tuning a model's learning rate

# Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.01,1.0]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_sequential_model((n_cols,), 100, [100], 2, 'relu', 'relu', 'softmax')
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr = lr)
    
    # Compile the model
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target)




###############################################################################
#Tell a model to stop when it isn't improving

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience = 2)

#Define model
model = get_sequential_model((n_cols,), 100, [100], 2, 'relu', 'relu', 'softmax')

#Compile model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target, validation_split=0.3, epochs = 30, callbacks = [early_stopping_monitor])





###############################################################################
# Plot validation score 


# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_1
model_1 = get_sequential_model((n_cols,), 150, [150,150], 2, 'relu', 'relu', 'softmax')

# Create the new model: model_2
model_2 = get_sequential_model((n_cols,), 100, [100], 2, 'relu', 'relu', 'softmax')



# Compile 
model_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=30, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=30, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# MNIST
df_mnist = pd.read_csv('C:\\Users\\KnightG\\Dropbox\\repos\\deep-learning-keras-Python\\mnist.csv', header = None) 
predictors_m = df_mnist.drop([0], axis=1).as_matrix()
target_m = to_categorical(df_mnist[0])
# extract shape of the predictors
n_cols_m = predictors_m.shape[1]

# Create the model: model
model = get_sequential_model((n_cols_m,), 250, [250,250,250,250,250,250,250], 10, 'relu', 'relu', 'softmax')


# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors_m,target_m, epochs=30, validation_split = 0.1, callbacks=[early_stopping_monitor])

