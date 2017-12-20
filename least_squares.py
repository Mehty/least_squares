import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from random import * 

# x-values
data_X = np.arange(1,101)

# y-values
data_Y = []
for i in range(1,101):
	data_Y.append(i + uniform(-5,6))

y = np.array(data_Y)

A = []
for i in data_X:
	A.append([i,1])
	
A = np.matrix(A)

# least_squares calculation
x = (np.dot(np.dot(inv(np.dot(A.transpose(), A)), A.transpose()), y)).getA()[0]
print x

# plotting the result
plt.plot(data_X,y,'ro')
plt.ylabel('y-values')
plt.xlabel('x-values')
plt.plot((x[0]*(data_X))+x[1])
plt.show()
