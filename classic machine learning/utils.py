import numpy as np 
from numpy.random  import random


def function_data_generator(function, n_points=1000, n=5, coefficients = None, error = 1E-1, bias = True):
	"""Generate coefficients, data matrix, and result points vector 
	from applying linear combination of coefficients time function(points)"""

	X = get_domain(n,n_points)
	
	a = coefficients if coefficients else generate_random_coefficients(n)
	b = bias*generate_random_coefficients(1)
	
	y = function(X).dot(a) + b
	y +=  random(n_points)*error*max(y)

	X = X if not bias else np.vstack((X.T, np.ones(X.shape[0]))).T
	a = a if not bias else np.hstack((a,b))

	return a, X, y

def get_domain(n, n_points):
	return random(size=(n_points,n))

def generate_random_coefficients(n):
	return random(n)*2-1 


def fair_partition(n,n_points):
	return (np.array([1/n for _ in range(n)])*n_points).astype(int)


def linear_regression(function,X,y):
	pseudo_inverse = get_pseudo_inverse(X)
	fitted_coefficients = pseudo_inverse.dot(y)

	return fitted_coefficients


def get_pseudo_inverse(X):
	return np.linalg.inv(X.T.dot(X)).dot(X.T)




def cumulative_partition(n, n_points):
	partitions = random(5)
	partitions /= sum(partitions)
	partitions *= n_points
	partitions = partitions.astype(int)
	rest = n_points - sum(partitions)
	partitions[0] += rest 
	
	return partitions

def generate_overlapped_normal_distribution(p=4, dimension=3, n_points=1000):

	means = np.array([random(dimension)*2-1 for _ in range(p)])
	max_mean_distance = max_distance_array(means)

	normal_distributions = []
	for mean in means:
		normal_distributions.append(np.random.normal(
			loc = mean,
			scale = max_mean_distance/10 , 
			size = (n_points,dimension),
			))
	return means,normal_distributions



def max_distance_array(array):

	pair_of_points = cartesian_product(array)
	norms = [np.linalg.norm(pts[0]-pts[1]) for pts in pair_of_points]
	max_norm = max(norms)

	return max_norm
	

def cartesian_product(array):
	"""thought for arrays nxm arrays"""

	combinations = []
	for i in range(array.shape[0]):
		for j in range(i,array.shape[0]):
			combinations.append((array[i],array[j]))

	return combinations


def between_class_matrix(X, y):
	labels = np.unique(y)
	idxs_for_labels = np.array([y==l for l in labels])
	cs = np.array([idx[idx==True].shape[0] for idx in idxs_for_labels])
	
	means_for_labels = np.array([np.mean(X[idx], axis=0).reshape(X.shape[1],1)
	 for idx in idxs_for_labels])
	mean_total = np.mean(X, axis=0).reshape(X.shape[1],1)
	
	SB_s = np.array([c_i*(mean_i-mean_total).dot((mean_i-mean_total).T) 
	for c_i,mean_i in zip(cs, means_for_labels)])

	return np.sum(SB_s, axis=0)


def _SWj(X_j, mean_j):
	
	SWj = 0
	dim = X_j.shape[1]	
	for xij in X_j:
		tmp_x = (xij-mean_j).reshape(dim,1)
		SWj += tmp_x.dot(tmp_x.T)
	
	return SWj


def within_class_matrix(X, y):
	labels = np.unique(y)
	idxs_for_labels = np.array([y==l for l in labels])

	means_for_labels = np.array([np.mean(X[idx], axis=0)
	 for idx in idxs_for_labels])
	

	SW_j = 0
	for idx, mean_j in zip(idxs_for_labels, means_for_labels):
		tmp_SWj = _SWj(X[idx], mean_j)
		SW_j += tmp_SWj

	return SW_j


def LDA_compression(X, y, l=1):
	SB = between_class_matrix(X, y)
	SW = within_class_matrix(X,y) 
	try:
		SW_1 = np.linalg.inv(SW)
	except:
		delta = 1000000000
		SW_1 = np.linalg.inv(SW+delta*np.identity(SW.shape[0]))
	
	W = SW_1.dot(SB)

	eval,evector = np.linalg.eig(W)
	evector = evector.real
	eval = eval.real

	sorted_idx = np.argsort(eval)[::-1]
	return np.vstack(evector[sorted_idx[:l],:])


def PCA_compression(X, l=1):
	A = X.T.dot(X)
	eval, evector = np.linalg.eig(A)
	evector = evector.real
	eval = eval.real

	sorted_idx = np.argsort(eval)[::-1]
	return np.vstack(evector[sorted_idx[:l],:])

	
def MSE(u,w):
	return np.sqrt(np.sum((u-w)**2)/u.shape[0])

def accuracy(real, predicted):	
	mask = (predicted == real)
	mask = mask[mask==True]
	acc = mask.shape[0]/real.shape[0]
	return acc

def logit(x):
    return 1/(1+np.e**(-x))

def dlogit(x):
	return np.e**(-x)/(((1+np.e**(-x)))**2)

def linear(xn):
	return xn

def quadratic(xn):
	return xn**2

def sin(xn):
	return np.sin(xn)

def cos(xn):
	return np.cos(xn)




FUNCTIONS = {
	"linear" : linear,
	"quadratic": quadratic,
	"sin": sin,
	"cos": cos
}


def main():
	pass


if __name__=='__main__':
	main()

