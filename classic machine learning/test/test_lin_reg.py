# Taking into account data coming from an N-dimensional field 
#that is {(x1,...,xn)_i} 
#implement a linear regression solution using least squares 
# in order to calculate parameters b,a_1,...,a_n. 


#The approach that will be taken is to take into account that
#I want to minimize the norm

from .. import utils as u


def main():

	a, X, y = u.function_data_generator(u.linear)
	fitted_coefficients = u.linear_regression(u.linear, X, y)
	
	error = u.MSE(X.dot(a),y)
	error_coefficients = u.MSE(a, fitted_coefficients)

	print("Mean Squared Error:", error)
	print("Mean Squared Error of Coefficients:", error_coefficients)


#things can undergo compression to the same domain, be kind of isomorphic there, or isomorphic between each
#of the compressed domains, define and talk about it there

if __name__=='__main__':
	main()