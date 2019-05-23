#!/usr/bin/env python3
import numpy as np 

# Set test to "Hello World" and print it.
test = "Hello World"
print("test: " + test)

## 1 - Building basic functions with numpy
# 1.1 - sigmoid function, np.exp()
def sigmoid(x):
	"""
	Compute sigmoid of x.

	Arguments:
	x -- A scalar

	Return:
	s -- sigmoid(x)
	"""

	s = 1 / (1 + np.exp(-x))

	return s


# 1.2 - Sigmoid gradient
def sigmoid_derivative(x):
	"""
	Compute the gradient (also called the slope or derivative) of
	the simoid function with respect to its input x.
	You can store the output of the sigmoid function into variables
	and then use it to calculate the gradient.

	Arguments:
	x -- A scalar or numpy array

	Return:
	ds -- Your computed gradient.
	"""

	s = sigmoid(x)
	ds = s * (1-s)

	return ds


# 1.3 - Reshaping arrays
def image2vector(image):
	"""
	Argument:
	image -- a numpy array of shape (length, height, depth)

	Returns:
	v -- a vector of shape (length*height*depth, 1)
	"""

	v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))

	return v


# 1.4 - Normalizing rows
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
 	
    # Compute x_norm as the norm 2 of x.
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    
    # Divide x by its norm.
    x = x/x_norm

    return x



# 1.5 - Broadcasting and the Softmax function
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    # Apply exp() element-wise to x.
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp.
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum
    
    return s


## 2 - Vectorization
# 2.1 - Implement the L1 and L2 loss functions
def L1(yhat, y):
	"""
	Arguments:
	yhat -- vector of size m (predicted labels)
	y -- vector of size m (true label)

	Returns:
	loss -- the value of the L1 loss function defined above.
	"""
	
	loss = np.sum(abs(yhat-y))

	return loss


def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    loss = np.dot((yhat-y).T, yhat-y)
    
    return loss

