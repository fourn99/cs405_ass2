# -*- coding: utf-8 -*-

"""
Question 3:
Implement the following two algorithms, from scratch, in python (using only the numpy import)
A) Gaussian Naive Bayes (probabilistic modeling)
B) Perceptron learning rule (Linear modeling) if perceptron does not converge run for 1000 iterations
do NOT copy-paste the sklearn code, or any other code from the internet (i will check this)


test your algorithms on the Linnerrud dataset using all 3 attributes, and only the chinups outcome,

rst de
ne new vector assigning binary classe to the outcome of chinups as follows:
if(chinups>median(chinups)) then chinups=0 else chinups=1

use these classes (0/1) to train the perceptron and build the probability table

QUESTION 3 OUTPUT: two .txt files:
gnb_results.txt => 20 probability values output by Gaussian Naive Bayes,
each value is P(chinups=1 | instance_i), where instance_i are the attributes of ith instance

perceptron_results.txt => 20 prediction values output by perceptron
each value is a weighted sum (dot product of perceptron’s weights with attribute values)

"""