# -*- coding: utf-8 -*-

"""
Question 2:
A) get the Linnerrud data using: data = load_linnerrud()
-weight, waist, and heartrate are attributes, chinups, situps, and jumps are outcomes
    
B) using numpy’s matrix functions (np.dot, np.transpose, etc.),
    compute the linear-least-squares solution, 
    ending the intercept and slope of best fit line for each [attribute, outcome] pair (attribute on x-axis, outcome on y-axis)
*hint - be sure to augment the attribute vectors with a column of 1’s (so LLS can 
nd the intercept)

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
    
display the slope and intercept of each scatter plot’ as the title of each scatter plot, 
as well as the attribute/outcome name on the x/y axis respectively
"""

from sklearn.datasets import load_linnerud
import numpy as np 
from numpy.linalg import inv 
import matplotlib.pyplot as plt 

data = load_linnerud()

raw_data = data.data
features_names = data.feature_names

target_data = data.targets
target_names = data.target_names



"""
# EXAMPLE FROM CLASS

# perform least square regression using matrix multiplication
W = np.dot(np.dot(inv((np.dot(X.T,X))),X.T),Y)


best-fit-line.py
xdata = np.zeros([10,2])
xdata[:,0] = 1
xdata[:,1] = np.arange(0,10)
ydata = xdata[:,1] + np.random.rand(10)*20
#plt.plot(xdata[:,1],ydata);
plt.plot(xdata[:,1],ydata,'o')

#xdata = np.expand_dims(xdata,axis=1)
ydata = np.expand_dims(ydata,axis=1)

# compute the weights
W = np.dot(np.dot(inv((np.dot(xdata.T,xdata))),xdata.T),ydata)

# plot the least square line 
plt.plot(xdata[:,1],np.dot(xdata,W)); plt.plot(xdata[:,1],ydata,'o')

# subtract mean from ydata and xdata
xdata[:,1] = xdata[:,1] - np.mean(xdata[:,1])
ydata = ydata - np.mean(ydata)

# get the correlation coefficient 
r = np.sum(xdata[:,1]*ydata[:,0]) \
    /np.sqrt(np.sum(xdata[:,1]*xdata[:,1]) \
    *np.sum(ydata*ydata))
"""




