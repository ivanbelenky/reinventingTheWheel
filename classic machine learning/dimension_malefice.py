import utils as u
import matplotlib.pyplot as plt
import numpy as np

import sys

plt.style.use('dark_background')


def main():
	"""malefice study"""
	num_of_points = int(sys.argv[1])
	function = u.FUNCTIONS[sys.argv[2]]


	dimensions = [_ for _ in range(2,100)]
	errors = []
	errors_coefficients = []
	stds = []
	biases = []
	for n in dimensions:
		a, X, y = u.function_data_generator(function, n_points=num_of_points,n=n)
		fitted_coefficients = u.linear_regression(u.linear, X, y)
		biases.append(a[-1])

		stds.append(np.std(X.dot(a)-y))
		errors.append(u.MSE(X.dot(a),y))
		errors_coefficients.append(u.MSE(a, fitted_coefficients))

	fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
	ax[0].plot(dimensions,errors,label='Mean Squared Errors')
	ax[0].plot(dimensions, errors_coefficients, label='Mean Squared Errors in Coefficients')
	ax[0].plot(dimensions, stds, label='Standard Deviation of Errors')
	
	ax[0].legend(loc=1)

	ax[1].plot(dimensions, biases, label='Biases')
	ax[1].legend(loc=1)
	
	plt.show()


if __name__ == '__main__':
	main()