import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    lwr = LocallyWeightedLinearRegression(tau=tau)
    lwr.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y_predict = lwr.predict(x_valid)
    mse = np.mean((y_predict - y_valid) ** 2)
    print(f'mse = {mse}')

    # Plot validation predictions on top of training set
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_valid, y_predict, 'ro', linewidth=2)
    plt.plot(x_valid, y_valid, 'gs', linewidth=2)

    # No need to save predictions
    # Plot data
    plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        def w_matrixs(obj_x):
            """

            Args:
                obj_x: Objective. Shape (m, n)

            Returns:

            """
            m, n = obj_x.shape
            w = np.exp(-1/(2 * (self.tau ** 2)) * (np.linalg.norm(self.x - obj_x.reshape((m, 1, n)), 2, axis=2) ** 2))
            return np.array([np.diag(raw) for raw in w])
        w_m = w_matrixs(x)
        thetas = np.linalg.inv((self.x.T @ w_m @ self.x)) @ self.x.T @ w_m @ self.y
        return np.diag(x @ thetas.T)
        # *** END CODE HERE ***


if __name__ == '__main__':
    main(tau=5e-2,
         train_path='../data/ds5_train.csv',
         eval_path='../data/ds5_valid.csv')
