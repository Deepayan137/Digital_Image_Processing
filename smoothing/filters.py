import numpy as np 

from math import e
def gaussian_filter(n_dim, sigma):
	gauss_filter = np.zeros((n_dim, n_dim))
	mat = np.zeros((n_dim,n_dim))
	normalization_constant = 1/(float(np.sqrt(2*np.pi))*2*sigma)
	pos = [-1, 0, 1]
	for i in range(n_dim):
		for j in range(n_dim):
			gauss_filter[j,i] = e**(-((pos[i]**2 + pos[j]**2)/float(2*sigma)))
			mat[j ,i] = -(pos[i]**2 + pos[j]**2)/float(2*sigma)	
			
			
	return gauss_filter*normalization_constant
print(gaussian_filter(3,1))