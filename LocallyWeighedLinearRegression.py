import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('ex1data.txt', header = None, names = ['Population', 'profits'])
data.sort_values('Population', inplace=True)
data.insert(0, 'Ones', 1)
cols = data.shape[1]
x = data.iloc[:, 0: cols-1]
y = data.iloc[:, cols-1: cols]
x = np.matrix(x.values)
y = np.matrix(y.values)
thetaChart = np.matrix(np.zeros(x.shape))
def computeWeights(testPoint, x, tau):
    m = x.shape[0]
    weights = np.matrix(np.eye(m))
    for i in range(m):
        difference = testPoint - x[i,:]
        weights[i,i] = np.exp(difference*difference.T/(-2*tau**2))
    return weights

def gradientDescent(x, y, thetaChart, alpha, iters, tau):
    param = thetaChart.shape[1]
    xNumber = x.shape[0]
    for i in range(iters):
        
        for j in range(xNumber):
            
            weights = computeWeights(x[j,:], x, tau)
            error = x*thetaChart[j,:].T - y
            
            for k in range(param):
                term = weights* np.multiply(error, x[:, k])
                thetaChart[j,k] = thetaChart[j,k] - (alpha/len(x))*np.sum(term)
    
    return thetaChart
alpha = 0.05
iters = 1000
tau = 1
finalChart = gradientDescent(x, y, thetaChart,alpha, iters, tau)
F = np.zeros(x.shape[0])
for i in range(x.shape[0]):
    F[i] = np.sum(np.multiply(thetaChart[i,:], x[i,:])) 
plt.figure(figsize = (12,8))
plt.scatter(data.Population, data.profits, label = 'training set')
plt.plot(data.Population, F, color = 'red',label = 'prediction')
plt.xlabel('Population')
plt.ylabel('profits')
plt.legend()
plt.show()
