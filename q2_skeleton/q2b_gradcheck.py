import random

import numpy as np
from numpy.testing import assert_allclose


def gradcheck_naive(f, x, gradient_text=""):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradient_text -- a string detailing some context about the gradient computation
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4         # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        x_plus_h = np.copy(x)
        x_minus_h = np.copy(x)

        x_plus_h[ix] += h
        x_minus_h[ix] -= h

        random.setstate(rndstate)
        fx_plus_h, _ = f(x_plus_h)

        random.setstate(rndstate)
        fx_minus_h, _ = f(x_minus_h)

        numgrad = (fx_plus_h - fx_minus_h) / (2 * h)

        assert_allclose(numgrad, grad[ix], rtol=1e-5,
                        err_msg=f"Gradient check failed for {gradient_text}.\n"
                                f"First gradient error found at index {ix} in the vector of gradients\n"
                                f"Your gradient: {grad[ix]} \t Numerical gradient: {numgrad}")

        it.iternext()

    print("Gradient check passed!")


def test_gradcheck_basic():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), 2*x)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))       # scalar test
    gradcheck_naive(quad, np.random.randn(3,))     # 1-D test
    gradcheck_naive(quad, np.random.randn(4, 5))   # 2-D test
    print()


def your_gradcheck_test():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    # Scalar input test
    quad = lambda x: (np.sum(x ** 2), 2 * x)
    x = np.array(123.456)
    gradcheck_naive(quad, x)

    # 1D array test
    quad = lambda x: (np.sum(x ** 2), 2 * x)
    x = np.random.randn(3)
    gradcheck_naive(quad, x)

    # 2D array test
    quad = lambda x: (np.sum(x ** 2), 2 * x)
    x = np.random.randn(4, 5)
    gradcheck_naive(quad, x)

    # Zero input test
    quad = lambda x: (np.sum(x ** 2), 2 * x)
    x = np.zeros((3, 3))
    gradcheck_naive(quad, x)

    # Large numbers test
    quad = lambda x: (np.sum(x ** 2), 2 * x)
    x = np.array([1e5, 1e5])
    gradcheck_naive(quad, x)

    # Custom function test
    def custom_func(x):
        fx = np.sum(np.sin(x))
        grad = np.cos(x)
        return fx, grad

    x = np.random.randn(5)
    gradcheck_naive(custom_func, x)

    # Invalid gradient test
    def wrong_grad(x):
        fx = np.sum(x ** 2)
        grad = x  # Wrong gradient (should be 2*x)
        return fx, grad

    x = np.array([1.0, 2.0])

    # Your gradcheck implementation test
    def cubic(x):
        fx = np.sum(x ** 3)
        grad = 3 * (x ** 2)
        return fx, grad

    x = np.random.randn(3)
    gradcheck_naive(cubic, x)

    # Multidimensional input test
    def matrix_func(x):
        fx = np.sum(x ** 2 + 2 * x)
        grad = 2 * x + 2
        return fx, grad

    x = np.random.randn(3, 3, 3)
    gradcheck_naive(matrix_func, x)

    # Edge cases test
    quad = lambda x: (np.sum(x ** 2), 2 * x)
    x = np.array([1e-10, 1e-10])  # Very small numbers
    gradcheck_naive(quad, x)


if __name__ == "__main__":
    test_gradcheck_basic()
    your_gradcheck_test()
