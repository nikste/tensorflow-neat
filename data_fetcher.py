"""generates training data"""


from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets.samples_generator import make_gaussian_quantiles
import numpy as np

def get_mnist():
    """retrieves tensorflow version of mnist"""
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist


def get_gaussian_quantiles(n_samples=1000):
    x, y = make_gaussian_quantiles(n_samples=n_samples, n_features=2, n_classes=2)
    y = np.asarray([[0., 1.] if y_ == 0 else [1., 0,] for y_ in y])

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x,y

def generate_xor(n_samples=1000):
    # num_samples,num_features
    X1 = np.random.randint(0,2,[n_samples,2])
    y = np.logical_xor(X1[:,0],X1[:,1]).astype(np.float32)
    y = np.asarray([[0,1] if y_ == 0 else [1,0] for y_ in y])
    return X1, y
