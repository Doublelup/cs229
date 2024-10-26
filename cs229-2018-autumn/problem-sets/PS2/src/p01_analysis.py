from p01_lr import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_a, y_a = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    x_b, y_b = util.load_csv('../data/ds1_b.csv', add_intercept=True)

    # plot dataset A
    plt.figure()
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.plot(x_a[y_a == 1, -2], x_a[y_a == 1, -1], 'go', linewidth=2)
    plt.plot(x_a[y_a == -1, -2], x_a[y_a == -1, -1], 'rx', linewidth=2)
    plt.show()

    # plot dataset B
    plt.figure()
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.plot(x_b[y_b == 1, -2], x_b[y_b == 1, -1], 'go', linewidth=2)
    plt.plot(x_b[y_b == -1, -2], x_b[y_b == -1, -1], 'rx', linewidth=2)
    plt.show()

