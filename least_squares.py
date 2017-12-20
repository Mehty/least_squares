import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from random import * 

# finds the best fit line using the least_square method 
def best_fit_line(data_X, data_Y):
	A = []
	for i in data_X:
		A.append([i,1])
	A = np.matrix(A)

	# least_squares calculation
	x = (np.dot(np.dot(inv(np.dot(A.transpose(), A)), A.transpose()), data_Y)).getA()[0]
	return x

# finds the best fit prabola using the least_square method 
def best_fit_prab(data_X, data_Y):
	A = []
	for i in data_X:
		A.append([i*i,i,1])
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

x = best_fit_line(data_X, data_Y)

x2_line = best_fit_line(np.array([-3, -2, 1, 2, 4]), np.array([10, 4, 0, 4, 15]))
x2_prab = best_fit_prab(np.array([-3, -2, 1, 2, 4]), np.array([10, 4, 0, 4, 15]))

# plotting the result
#plt.plot(data_X,data_Y,'ro')
plt.plot([-3, -2, 1, 2, 4], [10, 4, 0, 4, 15], 'ro')
plt.ylabel('y-values')
plt.xlabel('x-values')
#plt.plot(x[0]*data_X*data_X + x[1]*data_X + x[2])
data_X2 = np.arange(-10, 11)
data_Y2_prab = []
data_Y2_line = []
for x in data_X2:
	data_Y2_prab.append((x2_prab[0]*x*x) + (x2_prab[1]*x) + x2_prab[2])
	data_Y2_line.append((x2_line[0]*x) + x2_line[1])
data_Y2_prab = np.array(data_Y2_prab)
data_Y2_line = np.array(data_Y2_line)
plt.plot(data_X2, data_Y2_prab) 
plt.plot(data_X2, data_Y2_line, 'g-')
plt.show()
