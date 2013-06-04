import matplotlib.pyplot as plt
import numpy as np

# generating some uncorrelated data
data = np.random.rand(10,100) # each row of represents a variable

# creating correlation between the variables
# variable 2 is correlated with all the other variables
data[2,:] = np.sum(data,0)
# variable 4 is correlated with variable 8
data[4,:] = np.log(data[8,:])*0.5

# plotting the correlation matrix
R = np.corrcoef(data)

plt.pcolor(R)
plt.colorbar()
plt.yticks(np.arange(0.5,10.5), range(0,10))
plt.xticks(np.arange(0.5,10.5), range(0,10))
plt.show()