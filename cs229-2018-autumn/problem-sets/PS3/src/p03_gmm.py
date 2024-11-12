import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    m, n = x.shape
    selector = np.random.randint(0, K, m)
    mu = np.empty((K, n))
    sigma = np.empty((K, n, n))
    for i in range(0, K):
        sub = x[selector == i]
        sub_m, _ = sub.shape
        if sub_m == 0:
            mu[i] = np.zeros((n,))
            sigma[i] = np.diag(np.ones((n,)))
        else:
            mu[i] = np.mean(sub, axis=0)
            sigma[i] = 1 / sub_m * (sub - mu[i]).T @ (sub - mu[i])
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = 1 / K * np.ones((K,))
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = 1 / K * np.ones((m, K))
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        m, n = x.shape
        det_sigma = np.linalg.det(sigma) # shape (k,)
        x_sub_mu = x - mu.reshape((K, 1, n)) # shape (k, m, n)
        exp_partition = np.exp(-0.5 * np.einsum('kmn, knl, kml->km', x_sub_mu, np.linalg.inv(sigma), x_sub_mu)) # shape (k, m)
        w = np.einsum('k, km, k->mk', det_sigma ** -0.5, exp_partition, phi) #shape (m, k)
        w /= w.sum(axis=1)[:, None]

        # (2) M-step: Update the model parameters phi, mu, and sigma
        denominator = np.einsum('mk->k', w)
        phi = 1/m * denominator
        # calculate mu
        mu = np.einsum('mk, mn, k->kn', w, x, denominator ** -1)
        # calculate sigma
        sigma = np.einsum('ik, kim, kin, k->kmn', w, x_sub_mu, x_sub_mu, denominator ** -1) # shape(k, n, n)

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        x_sub_mu = x - mu.reshape((K, 1, n))
        exp_partition = np.exp(-0.5 * np.einsum('kmn, knl, kml->km', x_sub_mu, np.linalg.inv(sigma), x_sub_mu))
        det_sigma = np.linalg.det(sigma)
        p = ((2 * np.pi) ** (-n/2)) * np.einsum('k, km, k->m', det_sigma ** -0.5, exp_partition, phi)
        ll = np.sum(np.log(p))
        if prev_ll and ll < prev_ll:
            print('error')
            print(prev_ll, ' ', ll)
        it += 1
        # *** END CODE HERE ***
    print(it)
    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    m, n = x.shape
    m_tilde, _ = x_tilde.shape
    w_tilde = np.zeros((m_tilde, K))
    w_tilde[np.arange(m_tilde), z.reshape(m_tilde, ).astype(int)] = alpha
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        det_sigma = np.linalg.det(sigma) # shape (k,)
        x_sub_mu = x - mu.reshape((K, 1, n)) # shape (k, m, n)
        x_tilde_sub_mu = x_tilde - mu.reshape((K, 1, n))
        exp_partition = np.exp(-0.5 * np.einsum('kmn, knl, kml->km', x_sub_mu, np.linalg.inv(sigma), x_sub_mu)) # shape (k, m)
        w = np.einsum('k, km, k->mk', det_sigma ** -0.5, exp_partition, phi) #shape (m, k)
        w /= w.sum(axis=1)[:, None]
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # calculate phi
        denominator = np.einsum('mk->k', w) + np.einsum('mk->k', w_tilde)
        phi = (1 / (m + alpha * m_tilde)) * denominator
        # calculate mu
        partition1 = np.einsum('mk, mn->kn', w, x)
        partition2 = np.einsum('mk, mn->kn', w_tilde, x_tilde)
        mu = np.einsum('kn, k->kn', partition1+partition2, denominator ** -1)
        # calculate sigma
        partition1 = np.einsum('ik, kim, kin->kmn', w, x_sub_mu, x_sub_mu) # shape(k, n, n)
        partition2 = np.einsum('ik, kim, kin->kmn', w_tilde, x_tilde_sub_mu, x_tilde_sub_mu)
        sigma = np.einsum('kmn, k->kmn', partition1 + partition2, denominator ** -1)
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        x_sub_mu = x - mu.reshape((K, 1, n))
        exp_partition = np.exp(-0.5 * np.einsum('kmn, knl, kml->km', x_sub_mu, np.linalg.inv(sigma), x_sub_mu))
        det_sigma = np.linalg.det(sigma)
        p0 = ((2 * np.pi) ** (-n/2)) * np.einsum('k, km, k->mk', det_sigma ** -0.5, exp_partition, phi)
        p1 = p0.sum(axis=1)
        x_tilde_sub_mu = x_tilde - mu.reshape((K, 1, n))
        exp_partition = np.exp(-0.5 * np.einsum('kmn, knl, kml->km', x_tilde_sub_mu, np.linalg.inv(sigma), x_tilde_sub_mu))
        p0 = ((2 * np.pi) ** (-n/2)) * np.einsum('k, km, k->mk', det_sigma ** -0.5, exp_partition, phi)
        p2 = p0[np.arange(m_tilde), z.reshape(m_tilde,).astype(int)]
        ll = np.sum(np.log(p1)) + alpha * np.sum(np.log(p2))
        if prev_ll and ll < prev_ll:
            print('error')
            print(prev_ll, ' ', ll)
        it += 1
        # *** END CODE HERE ***
    print(it)
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        # main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
