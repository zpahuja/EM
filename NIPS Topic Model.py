# import libs
import numpy as np
import matplotlib.pyplot as plt
import sys
from math import log

# read data
file = open('data/docword.nips.txt', 'r')
data = [int(datum) for datum in file.read().split()]

D = data[0]
W = data[1]
NNZ = data[2]

data = data[3:]

# store data as numpy matrix
x = np.zeros((D, W))

for i in range(0, NNZ, 3):
    x[data[i]-1, data[i+1] - 1] = data[i+2]

J = 30 # number of topics/ clusters

# p corresponds to probability of word in a topic
p = np.ones((J, W))
p = 1.0/W * p

# pi corresponds to probability that document belongs to topic
pi = np.ones(J)
pi = 1.0/J * pi

# function to get w_i,j
def w(i, j):
    numerator = 1.0
    denominator = 1.0

    for l in range(J):
        for k in range(W):
            temp = p[l,k]**x[i,k]
            if l == j:
                numerator *= temp
            denominator *= temp
        denominator *= pi[l]

    return (numerator * pi[j])/ denominator

# E-Step computation
def expectation():
    Q = 0.0
    for i in range(D):
        print("expectation round", i)
        for j in range(J):
            Q += (log(pi[j]) + np.dot(x[i,], np.log(p[j,]))) * w(i,j)
    return Q

# M-Step
def max_p(j):
    numer = 0
    denom = 0
    for i in range(D):
        w_ij = w(i,j)
        numer += x[i,] * w_ij
        denom += np.sum(x[i,]) * w_ij
    return numer/denom

def max_pi(j):
    pi_j = 0
    for i in range(D):
        pi_j += w(i,j)
    return pi_j/ D

# EM
prev_expectation = sys.maxsize
t = 0

while True:
    e = expectation()
    if abs(e - prev_expectation) < 100:
        break
    prev_expectation = e
    for j in range(J):
        p[j,] = max_p(j)
        pi[j] = max_pi(j)
    print(t, e)
    t += 1
