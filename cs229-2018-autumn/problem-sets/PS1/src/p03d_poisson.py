import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    pr = PoissonRegression(step_size=lr)
    pr.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_predict = pr.predict(x_eval)
    print(f'variance = {np.linalg.norm(y_predict-y_eval, 2)}, contrast to average of y_eval: {np.mean(y_eval)}')
    np.savetxt(pred_path, y_predict, fmt='%.3f', delimiter=',\n')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def grad_l_theta(x, y, theta):
            """

            Args:
                x: Shape(m, n).
                y: Shape(m, 1).
                theta: Shape(n,).

            Returns:
                gradiant of l(theta). Shape (n, 1).
            """
            m, n = x.shape
            return x.T @ (y - np.exp(x @ theta.reshape((n, 1))))

        m, n = x.shape
        y_col_v = y.reshape((m, 1))
        theta = np.ones((n, 1))
        new_theta = np.zeros((n, 1))
        count = 0
        while np.linalg.norm(new_theta - theta) >= self.eps:
            count += 1
            theta = new_theta
            new_theta = theta + (self.step_size/m) * grad_l_theta(x, y_col_v, theta)
        self.theta = new_theta.reshape((n,))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(self.theta @ x.T)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-7,
        train_path='../data/ds4_train.csv',
        eval_path='../data/ds4_valid.csv',
        pred_path='output/p03d_pred.txt')
