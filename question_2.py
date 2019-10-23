# -*- coding: utf-8 -*-

"""
Question 2:
A) get the Linnerrud data using: data = load_linnerrud()
-weight, waist, and heartrate are attributes, chinups, situps, and jumps are outcomes
    
B) using numpy’s matrix functions (np.dot, np.transpose, etc.),
    compute the linear-least-squares solution, 
    ending the intercept and slope of best fit line for each [attribute, outcome] pair (attribute on x-axis, outcome on y-axis)
*hint - be sure to augment the attribute vectors with a column of 1’s (so LLS can find the intercept)

QUESTION 2 OUTPUT: a figure with a 3x3 grid of nine (9) subplots, each showing a scatter plot and best fit line:
    i) x=weight, y=chinups. 
    ii) x=weight, y=situps. 
    iii) x=weight, y=jumps.
    iv) x=waist, y=chinups. 
    v) x=waist, y=situps. 
    vi) x=waist, y=jumps.
    vii) x=heartrate,y=chinups. 
    viii)x=heartrate, y=situps. 
    ix) x=heartrate,y=jumps
    
display the slope and intercept of each scatter plot as the title 
as well as the attribute/outcome name on the x/y axis respectively
"""

from sklearn.datasets import load_linnerud
import numpy as np 
from numpy.linalg import inv 
import matplotlib.pyplot as plt 


def get_weights_linear_regression(x, y):
    
    x_data = np.zeros([20,2])
    x_data[:,0] = 1
    x_data[:,1] = x
    y_data = y
    
    y_data = np.expand_dims(y_data,axis=1)
    
    # compute the weights
    W = np.dot(np.dot(inv((np.dot(x_data.T,x_data))),x_data.T),y_data)
        
    return W

# -- Get data
    
data_set = load_linnerud()

raw_data = data_set.data # Chins, Situps, Jumps
features_names = data_set.feature_names

target_data = data_set.target  # Weight, Waist, Pulse
target_names = data_set.target_names

fig, axis  = plt.subplots(3, 3)
fig.set_size_inches(20,25)

for i in range(len(target_names)):
    x_temp = target_data[:,i] 
    
    
    for j in range(len(features_names)):
        y_temp = raw_data[:,j]
        
        weights = get_weights_linear_regression(x_temp, y_temp)  # calculate weights
        intercept = float(weights[0])
        slope = float(weights[1])
         
        x_line = np.linspace(int(np.min(x_temp)), int(np.max(x_temp)), int(np.max(y_temp))) # set x domain
        y_line = slope*x_line + intercept # line equation
        axis[i,j].plot(x_temp, y_temp,'o')
        axis[i,j].plot(x_line, y_line)
        axis[i,j].set_title(target_names[i] + " vs " + features_names[j] + "-- Intercept: " + str(round(intercept, 3)) +" Slope: "+ str(round(slope,3)))
        axis[i,j].set(xlabel = target_names[i], ylabel = features_names[j])
    
plt.show()


