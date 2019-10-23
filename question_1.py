# -*- coding: utf-8 -*-


"""
Question 1:
A) using numpy, initialize an array of random numbers each number ranging between 0 and 1
-array should have shape=[1000,50] (1000 rows, 50 columns)

B) create the correlation matrix of pearson correlations between all pairs of rows from (1A)
- correlation matrix should have shape=[1000,1000])

C) using matplotlib, plot a 100-bin histogram, using values from lower triangle 
    of 1000x1000 correlation coefficient(r-values) matrix obtained in 1B 
    (omit the diagonal and all cells above the diagonal)
    
*hint - the histogram will be shaped like a gaussian

using the histogram, estimate the probability of obtaining an 
r-value > 0.75 or < -0.75 from correlating two random vectors of size 50. 

repeat A-C with only 10 columns in (A), 
how does the smaller sample affect the histogram in (C)?


QUESTION 1 OUTPUT: a figure with two histograms, 
    hist1 based on correlations of vectors of size 50,
    hist2 based on correlations of vectors of size 10. 
    display the probability from (C) as the title of the histograms
"""


import matplotlib.pyplot as plt
import numpy as np


def compute_correlation(x, y):
    
    x_adjusted = x - np.mean(x)
    y_adjusted = y - np.mean(y)
    
    r = np.sum(x_adjusted*y_adjusted)/np.sqrt(np.sum(x_adjusted*x_adjusted)*np.sum(y_adjusted*y_adjusted))
    
    return r


def answer_number_one():

    # -- Generate random data
    data_rand_50 = np.random.random((1000,50))  # initiate random array 1000x50
    
    # -- Compute correlation matrix
    
    # initiate empty array
    correlation_matrix_50 = [[0 for x in range(1000)] for y in range(1000)]
    
    # fill correlation matrix with pearson coefficient
    for i in range(1000):
       for j in range(1000):
            correlation_matrix_50[i][j] = compute_correlation(data_rand_50[i], data_rand_50[j])
            
#    # test correlation matrix made by us to the numpy algorithm
#    coefficent_matrix_50 = np.corrcoef(data_rand_50)  # compute correlation matrix
#    print(np.allclose(correlation_matrix_50, coefficent_matrix_50))
    
    # keep lower triangle of matrix, put in 1d array
    correlation_matrix_50 = np.asarray(np.ndarray.flatten(np.tril(correlation_matrix_50, k = -1)))
    correlation_matrix_50 = correlation_matrix_50[correlation_matrix_50 != 0] # remove 0s
    
    values, bins, patches = plt.hist(correlation_matrix_50, bins=100)  # plot 100 bins histogram
    
    nb_lower_50 = (correlation_matrix_50 < -.75).sum()
    nb_higher_50 = (correlation_matrix_50 > 0.75).sum()
    
    prob_50 = round(((nb_higher_50 + nb_lower_50) / len(correlation_matrix_50)) * 100, 3)
    
    # making the histogram look good
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(' 50 Column Histogram - P(X > 0.75, X < -.75) = '+str(prob_50)+ '%') # title should be p(X>0.75)
    
    plt.show()
    
    
    plt.clf()  # clear plots for new historgram
    
    #---------- 10 columns 
    
     # -- Generate random data
    data_rand_10 = np.random.random((1000,10))  # initiate random array 1000x50
    
    # -- Compute correlation matrix
    
    # initiate empty matrix
    correlation_matrix_10 = [[0 for x in range(1000)] for y in range(1000)]
    
    # fill correlation matrix with pearson coefficient
    for i in range(1000):
       for j in range(1000):
            correlation_matrix_10[i][j] = compute_correlation(data_rand_10[i], data_rand_10[j])
            
#    # test correlation matrix made by us to the numpy algorithm
#    coefficent_matrix_50 = np.corrcoef(data_rand_50)  # compute correlation matrix
#    print(np.allclose(correlation_matrix_50, coefficent_matrix_50))
    
    # keep lower triangle of matrix, put in 1d array
    correlation_matrix_10 = np.asarray(np.ndarray.flatten(np.tril(correlation_matrix_10, k = -1)))
    correlation_matrix_10 = correlation_matrix_10[correlation_matrix_10 != 0] # remove 0s
    
    values, bins, patches = plt.hist(correlation_matrix_10, bins=100)  # plot 100 bins histogram
    
    nb_lower_10 = (correlation_matrix_10 < -.75).sum()
    nb_higher_10 = (correlation_matrix_10 > 0.75).sum()
    
    prob_10 = round(((nb_higher_10 + nb_lower_10) / len(correlation_matrix_10)) * 100, 3)
    
    # making the histogram look good
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(' 10 Column Histogram - P(X > 0.75, X < -.75) = '+str(prob_10)+ '%') # title should be p(X>0.75)
    
    plt.show()








