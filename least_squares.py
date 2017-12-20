import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from random import * 

# finds the best fit line using the least_square method 
def best_fit(data_X, data_Y):
	A = []
	for i in data_X:
		A.append([i,1])
	A = np.matrix(A)

	# least_squares calculation
	x = (np.dot(np.dot(inv(np.dot(A.transpose(), A)), A.transpose()), data_Y)).getA()[0]
	return x


# x-values
data_X = np.arange(1,101)

# y-values
data_Y = []
for i in range(1,101):
	data_Y.append(i + uniform(-5,6))
data_Y = np.array(data_Y)

x = best_fit(data_X, data_Y)

# plotting the result
plt.plot(data_X,data_Y,'ro')
plt.ylabel('y-values')
plt.xlabel('x-values')
plt.plot((x[0]*(data_X))+x[1])
plt.show()
