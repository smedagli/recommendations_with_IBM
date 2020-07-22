import tqdm
import numpy as np
import pandas as pd
from itertools import product

from recommendations_with_IBM.read_data import reader


def svd(user_matrix: pd.DataFrame, k: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """ Performs the singular value decomposition (SVD)
    Args:
        user_matrix: shape (M, N); M users, N items
        k: number of latent features
    Returns:
        u: shape (M, M)
        s: shape (N, N) (default return of svd by numpy is an array with shape (N, 1) )
        vt: shape (N, N)
    """
    u, s, vt = np.linalg.svd(user_matrix)
    if k >= user_matrix.shape[1]:
        s_new = np.diag(s)
        return u, s_new, vt
    else:
        u_new = u[:, : k]
        s_new = np.diag(s[: k])
        v_new = vt[: k, :]
        return u_new, s_new, v_new


def estimate_from_svd(u: np.ndarray, s: np.ndarray, vt: np.ndarray) -> np.ndarray:
    """ Performs the estimation from the SVD components
    Args:
        u: U matrix for SVD
        s: sigma matrix for SVD (base vectors are sorted from most to least important)
        vt: V transpose matrix for SVD
    Returns:
        estimation of the matrix
    """
    return np.around(np.dot(np.dot(u, s), vt))


def compute_estimation_error(estimated, actual) -> float:
    """ Returns the error in estimation
    Args:
        estimated: estimated matrix with SVD (estimate_from_SVD)
        actual: "ground-truth"
    Returns:
        the error computed (not normalized)
    """
    diffs = np.subtract(actual, estimated)
    return np.sum(np.sum(np.abs(diffs)))


def FunkSVD(user_matrix: pd.DataFrame, latent_features=10, learning_rate=0.0001, iters=100):
    """ This function performs matrix factorization using a basic form of FunkSVD with no regularization
    Args:
        user_matrix: a matrix or DataFrame with users as rows, movies as columns, and ratings as values
        latent_features: (int) the number of latent features used
        learning_rate: (float) the learning rate
        iters: (int) the number of iterations
    Returns:
        u: (numpy array) a user by latent feature matrix
        vt: (numpy array) a latent feature by items matrix
        s: list of MSE (one for iteration)
    """
    mat = np.array(user_matrix)
    n_users, n_items = mat.shape
    num_read = np.count_nonzero(mat)

    # initialize the user and movie matrices with random values
    u = np.random.rand(n_users, latent_features)  # user matrix filled with random values of shape user x latent
    vt = np.random.rand(latent_features, n_items)  # item matrix filled with random values of shape latent x movies

    # initialize sse at 0 for first iteration
    s = []

    # header for running results
    # print("Optimization Statistics")
    # print("Iterations | Mean Squared Error ")

    # for each iteration
    iterations = range(iters)
    with tqdm.tqdm(iterations) as pbar:
        for _ in iterations:
            sse_accum = 0
            for i, j in product(range(n_users), range(n_items)):

                if mat[i, j] > 0:
                    e = mat[i, j] - np.dot(u, vt)[i, j]
                    sse_accum += e ** 2
                    for f in range(latent_features):
                        u[i, f] += learning_rate * 2 * (e * vt[f, j])
                        vt[f, j] += learning_rate * 2 * (e * u[i, f])
            pbar.update(1)
            s.append(sse_accum / num_read)
            # print("%d \t\t %f" % (iteration + 1, s[-1]))

    return u, vt, s

if __name__ == '__main__':
    u = reader.UserList(reduced=True)
    user_matrix = u.user_matrix
    user_mat, item_mat, s = FunkSVD(user_matrix)

    r = pd.DataFrame(np.dot(user_mat, item_mat))