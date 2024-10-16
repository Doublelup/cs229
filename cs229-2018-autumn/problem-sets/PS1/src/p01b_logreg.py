import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_predict = lr.predict(x_eval)
    np.savetxt(pred_path, y_predict, fmt='%d', delimiter=',\n')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.
#
    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
#
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.
#
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
#
        def h(theta, x):
            return 1 / (1 + np.exp(- theta.T @ x))
#
        m = x.shape[0]
        n = x.shape[1]
        self.theta = np.ones((n, 1))
        new_theta = np.zeros((n, 1))
        y.resize((m, 1))
#
        counter = 0
        while np.linalg.norm(self.theta - new_theta, ord=1) >= self.eps and counter < self.max_iter:
            counter += 1
            self.theta = new_theta
            g_eta = 1 / (1 + np.exp(-x @ self.theta))
            grad_j_theta = -1/m * x.T @ (y - g_eta)
            hessian_j_theta = (1/m * x.T @ np.diag(((1 - g_eta) * g_eta).flatten()) @ x)
            new_theta = self.theta - np.linalg.inv(hessian_j_theta) @ grad_j_theta
        self.theta = new_theta
        y.resize((m,))
        # *** END CODE HERE ***
#
    def predict(self, x):
        """Make a prediction given new inputs x.
#
        Args:
            x: Inputs of shape (m, n).
#
        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        y = x @ self.theta >= 0
        y.resize((x.shape[0],))
        return y
        # *** END CODE HERE ***

if __name__ == '__main__':
    # main(train_path='../data/ds1_train.csv',
    #      eval_path='../data/ds1_valid.csv',
    #      pred_path='output/p01b_pred_1.txt')

    x_train, y_train = util.load_dataset('../data/ds1_train.csv', add_intercept=True)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    accuracy1 = np.mean(lr.predict(x_train) == y_train)
    util.plot(x_train, y_train, lr.theta, save_path='../src/output/p01b_1.png')

    x_train, y_train = util.load_dataset('../data/ds2_train.csv', add_intercept=True)
    lr.fit(x_train, y_train)
    accuracy2 = np.mean(lr.predict(x_train) == y_train)
    util.plot(x_train, y_train, lr.theta, save_path='../src/output/p01b_2.png')
    print(f'accuracy on dataset1: {accuracy1}, on dataset2: {accuracy2}')

