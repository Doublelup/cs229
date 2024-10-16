import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    lr = LogisticRegression()
    # Part (c): Train and test on true labels
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    lr.fit(x_train, t_train)
    print(f'accuracy: {np.mean(lr.predict(x_train) == t_train)}')
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    np.savetxt(pred_path_c, lr.predict(x_test), fmt='%d', delimiter='\n')
    util.plot(x_test, t_test, lr.theta, save_path='output/p02c_plot.png')
    # Make sure to save outputs to pred_path_c

    # Part (d): Train on y-labels and test on true labels
    x_train, y_train = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    _, t_train = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    lr.fit(x_train, y_train)
    print(f'accuracy: {np.mean(lr.predict(x_train) == y_train)}')
    x_test, _ = util.load_dataset(test_path, label_col='y', add_intercept=True)
    np.savetxt(pred_path_d, lr.predict(x_test), fmt='%d', delimiter='\n')
    util.plot(x_train, t_train, lr.theta, save_path='output/p02d_plot.png')
    # Make sure to save outputs to pred_path_d

    # Part (e): Apply correction factor using validation set and test on true labels
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    _, t_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    labeled_set_x = x_valid[y_valid == 1]
    h_x = 1 / np.exp(-labeled_set_x @ lr.theta)
    alpha = 1/labeled_set_x.shape[0] * np.sum(h_x)
    def new_h(alpha, old_theta, x):
        old_probability = 1 / (1 + np.exp(-x @ old_theta))
        new_probability = old_probability / alpha
        return (new_probability >= 0.5).ravel()
    np.savetxt(pred_path_e, new_h(alpha, lr.theta, x_test), fmt='%d', delimiter='\n')
    print(f'accuracy: {np.mean(new_h(alpha, lr.theta, x_valid) == t_valid)}')
    new_theta = lr.theta + np.log(2 / alpha - 1) * np.array([[1], [0], [0]])
    util.plot(x_test, t_test, new_theta, save_path='output/p02e_plot.png')
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE


if __name__ == '__main__':
    main(train_path='../data/ds3_train.csv',
        valid_path='../data/ds3_valid.csv',
        test_path='../data/ds3_test.csv',
        pred_path='output/p02X_pred.txt')
