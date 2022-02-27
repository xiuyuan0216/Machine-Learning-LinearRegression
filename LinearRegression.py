import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('ex1data.txt', header = None, names = ['Population','profits'])

def computeCost(x, y, theta):
    inner = np.power(((x * theta.T)-y), 2)
    return np.sum(inner) / (2*len(x))

data.insert(0,'Ones',1)
cols = data.shape[1]
x = data.iloc[:, 0: cols-1]
y = data.iloc[:, cols-1: cols]
x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

def gradientDescent(x, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    param = int(theta.shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (x * theta.T) - y
        
        for j in range(param):
            term = np.multiply(error, x[:, j])
            print(term.shape)
            temp[0, j] = theta[0,j] - ((alpha/ len(x)))*np.sum(term)
            
        theta = temp
        cost[i] = computeCost(x, y, theta)
    
    return theta, cost

alpha = 0.01
iters = 1000
g, cost = gradientDescent(x, y, theta, alpha, iters)
X = np.linspace(data.Population.min(), data.Population.max(), 100)
F = g[0,0] + g[0,1]*X

plt.figure(figsize = (12,8))
plt.scatter(data.Population, data.profits, label = 'training set')
plt.plot(X, F, color = 'red', label = 'prediction')
plt.legend()
plt.xlabel('Population')
plt.ylabel('profits')
plt.show()

