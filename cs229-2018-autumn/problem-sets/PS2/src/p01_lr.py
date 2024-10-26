# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))
    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        theta = theta
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            util.plot(X, Y, theta, None)
            break
        if i % 2000000 == 0:
            util.plot(X, Y, theta, None)
            print(theta / prev_theta)
            print(theta - prev_theta)
    return


def main():
    # print('==== Training model on data set A ====')
    # Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    # logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    mean = 0
    variance = 0.04
    noise = np.random.normal(mean, variance, Xb.shape)
    disturbed_xb = Xb + noise
    logistic_regression(disturbed_xb, Yb)


if __name__ == '__main__':
    main()
