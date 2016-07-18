from data_fetcher import get_gaussian_quantiles, generate_xor
from standard_neat import start_neuroevolution
from tensorflow_utils import build_and_test
import numpy as np
# from standard_net import build_and_test





# print X.shape
# print y
# print y_test

# X, y = generate_xor(n_samples=1000)
# X_test, y_test = generate_xor(n_samples=10)
#
#
# build_and_test(X, y, X_test, y_test)

#    output
#   /      \
# hidden1 hidden2
#  |    x   |
# input1 input2



# assumption, nodes can only connect to nodes with higher numbers.
# take care for the case that genotype is shorter than connections!!!
# x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
# y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)

# x_test = x.copy()
# y_test = y.copy()
x, y = get_gaussian_quantiles(n_samples=100)
x_test, y_test = get_gaussian_quantiles(n_samples=100)


print x.shape
print y.shape
start_neuroevolution(x, y, x_test, y_test)

# build_and_test(connections, genotype, x, y, x_test, y_test)
