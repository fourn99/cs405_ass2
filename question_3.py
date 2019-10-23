# -*- coding: utf-8 -*-

"""
Question 3:
Implement the following two algorithms, from scratch, in python (using only the numpy import)

A) Gaussian Naive Bayes (probabilistic modeling)
    
B) Perceptron learning rule (Linear modeling) if perceptron does not converge run for 1000 iterations
do NOT copy-paste the sklearn code, or any other code from the internet (i will check this)


test your algorithms on the Linnerrud dataset using all 3 attributes, and only the chinups outcome,

first define new vector assigning binary classe to the outcome of chinups as follows:
if(chinups>median(chinups)) then chinups=0 else chinups=1

use these classes (0/1) to train the perceptron and build the probability table

QUESTION 3 OUTPUT: two .txt files:
gnb_results.txt => 20 probability values output by Gaussian Naive Bayes,
each value is P(chinups=1 | instance_i), where instance_i are the attributes of ith instance

perceptron_results.txt => 20 prediction values output by perceptron
each value is a weighted sum (dot product of perceptron’s weights with attribute values)

"""


from sklearn.datasets import load_linnerud
import numpy as np 

def compute_probs_gaussian(x, mean, var):
    
    return (np.exp(-(np.square(x-mean)/(2 * np.square(var)))))/(np.sqrt(2*np.pi)*var)


    

def gnb(attributes, outcome):
    
    nb_outcome_yes = (outcome == 1).sum()
    nb_outcome_no = len(outcome) - nb_outcome_yes
    
    prior_yes = nb_outcome_yes / len(outcome)
    prior_no = nb_outcome_no / len(outcome)
    
  
    # build tempory matrix with outcomes and attributes
    temp_attributes_outcomes = np.zeros((20,4))
    
    temp_attributes_outcomes[:,0] = attributes[:,0]
    temp_attributes_outcomes[:,1] = attributes[:,1]
    temp_attributes_outcomes[:,2] = attributes[:,2]
    temp_attributes_outcomes[:,3] = outcome
    
    list_mean_yes = []
    list_var_yes = []
    list_mean_no = []
    list_var_no = []
    

    # get mean and variance for each attributes, for yes and no
    for i in range(3): # for each column
        
        list_mean_yes.append(temp_attributes_outcomes[temp_attributes_outcomes[:,3]==1, i].mean())
        list_var_yes.append(temp_attributes_outcomes[temp_attributes_outcomes[:,3]==1, i].var())
        list_mean_no.append(temp_attributes_outcomes[temp_attributes_outcomes[:,3]==0, i].mean())
        list_var_no.append(temp_attributes_outcomes[temp_attributes_outcomes[:,3]==0, i].var())
    
#    matrix_probability_yes = np.zeros((20,3))
#    
#    for i in range(len(matrix_probability_yes)):
#        
#        for j in range(len(matrix_probability_yes[0])):
#            
#            temp_x = attributes[i][j]
#            temp_mean = list_mean[j]
#            temp_var = list_var[j]
#            matrix_probability_yes[i][j] = compute_probs_gaussian(temp_x, temp_mean, temp_var)
            
#    likelihood yes = f(weight i |chin=1)f(Waist i|chin=1)f(Pulse i|chin=1)prior_yes
#    likelihood no = f(weight i |chin=0)f(Waist i|chin=0)f(Pulse i|chin=0)prior_no
#    P(Chins = 1 |Weight i, Waist i,Pulse i) = Likelihood yes/ (Likelihood yes + Likelihood no) 
    
  
            
         
    print(list_mean_yes)
    print(list_mean_no)
    print(list_var_yes)
    print(list_var_no)
    
             
             
             
    
    
    
    return 1
        

        
    

# -- Get Data

data_set = load_linnerud()

chins_data = data_set.data[:,0] # Chins
attribute_data = data_set.target  # Weight, Waist, Pulse


# set chin ups target as 1 or 0 
chins_median = np.median(chins_data)
chins_target = np.zeros(20)

for i in range(len(chins_data)):
    
    if chins_data[i] <= chins_median:
        chins_target[i] = 1
        
# compute P(Chins = 1 | Weight, Waist, Pulse)
gnb(attribute_data, chins_target)









