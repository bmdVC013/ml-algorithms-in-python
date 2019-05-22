#!usr/bin/env python3
import numpy as np

# random 5-vector from the "standard normal" disrtibution 
a = np.random.randn(5) # DON'T use
print(a)
# display the shape of a
print(a.shape)

# display the transpose of a
print(a.T)

# a * a.T -> scalar
# (5,) * (,5) -> scalar
print(np.dot(a, a.T))
print()

# create a comlumn vector
a = np.random.randn(5,1) # SHOULD use
print(a)
print(a.T)
# (5,1) * (1,5) -> (5,5)
print(np.dot(a, a.T))

# to make sure shape of a, you should use
assert(a.shape == (5,1))
print('True')