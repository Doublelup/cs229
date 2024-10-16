import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    gda = GDA()
    gda.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_predict = gda.predict(x_eval)
    np.savetxt(pred_path, y_predict, fmt='%d', delimiter='\n')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y.resize((m, 1))
        y_sum_of_true = np.sum(y == 1)
        y_sum_of_false = np.sum(y == 0)
        phi = 1 / m * y_sum_of_true
        mu_0 = np.sum(x[y.reshape((m,)) == 0], axis=0).reshape((n, 1)) / y_sum_of_false
        mu_1 = np.sum(x[y.reshape((m,)) == 1], axis=0).reshape((n, 1)) / y_sum_of_true
        u = x.T - mu_0 @ (1 - y.T) - mu_1 @ y.T
        sigma = 1 / m * u @ u.T
        self.theta = phi, sigma, mu_0, mu_1
        y.resize((m,))
        theta_t = (mu_1 - mu_0).T @ np.linalg.inv(sigma)
        theta0 = -(np.log((1 - phi) / phi)) - 1/2 * (mu_1 - mu_0).T @ np.linalg.inv(sigma) @ (mu_0 + mu_1)
        self.theta = np.insert(theta_t, 0, theta0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, _ = x.shape
        x_full = np.hstack((np.ones((m, 1)), x))
        y = self.theta @ x_full.T >= 0
        return y
        # *** END CODE HERE

if __name__ == '__main__':
    # main(train_path='../data/ds1_train.csv',
    #      eval_path='../data/ds1_valid.csv',
    #      pred_path='output/p01e_pred_1.txt')

    x_train, y_train = util.load_dataset('../data/ds1_train.csv', add_intercept=False)
    x_valid, y_valid = util.load_dataset('../data/ds1_valid.csv', add_intercept=False)
    gda = GDA()
    gda.fit(x_train, y_train)
    util.plot(x_valid, y_valid, gda.theta, save_path='../src/output/p01e_1.png')
    accuracy1 = np.mean(gda.predict(x_valid) == y_valid)

    x_train, y_train = util.load_dataset('../data/ds2_train.csv', add_intercept=False)
    x_valid, y_valid = util.load_dataset('../data/ds2_valid.csv', add_intercept=False)
    gda.fit(x_train, y_train)
    util.plot(x_valid, y_valid, gda.theta, save_path='../src/output/p01e_2.png')
    accuracy2 = np.mean(gda.predict(x_valid) == y_valid)

    print(f'accuracy on dataset1: {accuracy1}, on dataset2: {accuracy2}')
