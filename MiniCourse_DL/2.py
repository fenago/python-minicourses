import theano
from theano import tensor

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# declare variables
x, y = tensor.dmatrices('x', 'y')

# create simple expression for each operation
diff = x - y

abs_diff = abs(diff)
diff_squared = diff**2

# convert the expression into callable object
f = theano.function([x, y], [diff, abs_diff, diff_squared])

# call the function and store the result in a variable
result= f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

# format print for readability
print('Difference: ')
print(result[0])

print('Absolute Difference: ')
print(result[1])

print('Squared Difference: ')
print(result[2])