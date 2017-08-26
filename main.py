from data_fetcher import get_gaussian_quantiles, generate_xor
from standard_neat import start_neuroevolution



# X, y = generate_xor(n_samples=1000)
# X_test, y_test = generate_xor(n_samples=10)


x, y = generate_xor(n_samples=100)
x_test, y_test = generate_xor(n_samples=100)


print x.shape
print y.shape
start_neuroevolution(x, y, x_test, y_test)

# build_and_test(connections, genotype, x, y, x_test, y_test)
