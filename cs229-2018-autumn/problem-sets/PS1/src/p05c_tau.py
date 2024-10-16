import matplotlib.pyplot as plt
import numpy as np
import util
import math

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    results = []
    lwr = LocallyWeightedLinearRegression(tau=0)
    lwr.fit(x_train, y_train)
    def plot(y_predict, tau):
        plt.figure()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x_train, y_train, 'bx', linewidth=2)
        plt.plot(x_valid, y_predict, 'ro', linewidth=2)
        plt.text(4, 1, f'tau = {tau}', fontsize=12, color='black')
        plt.show()

    for tau in tau_values:
        lwr.tau = tau
        print(f'tau = {lwr.tau}')
        y_predict = lwr.predict(x_valid)
        mse = np.mean((y_valid - y_predict) ** 2)
        results.append((tau, mse))
        plot(y_predict, tau)

    # Fit a LWR model with the best tau value
    best_pair = (-1, math.inf)
    for pair in results:
        if best_pair[1] > pair[1]:
            best_pair = pair
    print(f'Best tau is {best_pair[0]}, mse = {best_pair[1]}')

    lwr.tau = best_pair[0]

    # Run on the test set to get the MSE value
    y_predict = lwr.predict(x_test)
    mse = np.mean((y_test - y_predict) ** 2)
    print(f'The mse of applying best tau({lwr.tau}) on test set is {mse}.')

    # Save predictions to pred_path
    np.savetxt(pred_path, y_predict, fmt='%.3f', delimiter='\n')
    # Plot data
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_test, y_predict, 'ro', linewidth=2)
    plt.plot(x_test, y_test, 'gs', linewidth=2)
    plt.text(4, 1, f'tau = {lwr.tau}', fontsize=12, color='black')
    plt.show()
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='../data/ds5_train.csv',
         valid_path='../data/ds5_valid.csv',
         test_path='../data/ds5_test.csv',
         pred_path='output/p05c_pred.txt')
