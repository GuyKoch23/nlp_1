import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        x_max = np.max(x, axis=1, keepdims=True)
        x = x - x_max
        exp_x = np.exp(x)
        x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        exp_x = np.exp(x)
        x = exp_x / np.sum(exp_x)

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([[0.26894142, 0.73105858], [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    # Test 1: Basic vector input
    test_vector = np.array([1.0, 2.0, 3.0])
    expected_vector = np.array([0.09003057, 0.24472847, 0.66524096])
    assert np.allclose(softmax(test_vector), expected_vector, rtol=1e-05, atol=1e-06)

    # Test 2: Vector with large numbers (testing numerical stability)
    test_large = np.array([1000.0, 2000.0, 3000.0])
    expected_large = np.array([0.0, 0.0, 1.0])
    assert np.allclose(softmax(test_large), expected_large, rtol=1e-05, atol=1e-06)

    # Test 3: Vector with negative numbers
    test_negative = np.array([-1.0, -2.0, -3.0])
    expected_negative = np.array([0.66524096, 0.24472847, 0.09003057])
    assert np.allclose(
        softmax(test_negative), expected_negative, rtol=1e-05, atol=1e-06
    )

    # Test 4: Matrix input
    test_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected_matrix = np.array(
        [[0.09003057, 0.24472847, 0.66524096], [0.09003057, 0.24472847, 0.66524096]]
    )
    assert np.allclose(softmax(test_matrix), expected_matrix, rtol=1e-05, atol=1e-06)

    # Test 5: Matrix with varying scales
    test_mixed = np.array([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0], [-1.0, -2.0, -3.0]])
    expected_mixed = np.array(
        [
            [0.09003057, 0.24472847, 0.66524096],
            [0.0, 0.0, 1.0],
            [0.66524096, 0.24472847, 0.09003057],
        ]
    )
    assert np.allclose(softmax(test_mixed), expected_mixed, rtol=1e-05, atol=1e-06)

    # Test 6: Single element vector
    test_single = np.array([1.0])
    expected_single = np.array([1.0])
    assert np.allclose(softmax(test_single), expected_single, rtol=1e-05, atol=1e-06)

    # Test 7: Zero vector
    test_zero = np.zeros(3)
    expected_zero = np.array([1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(softmax(test_zero), expected_zero, rtol=1e-05, atol=1e-06)

    # Test 8: Shape preservation
    test_shape = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result_shape = softmax(test_shape)
    assert result_shape.shape == test_shape.shape

    # Test 10: 3D array (should maintain original shape)
    test_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result_3d = softmax(test_3d)
    assert result_3d.shape == test_3d.shape


if __name__ == "__main__":
    test_softmax_basic()
    your_softmax_test()
