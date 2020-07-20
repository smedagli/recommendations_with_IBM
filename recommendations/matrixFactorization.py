import tqdm
import numpy as np
import pandas as pd
from itertools import product

from recommendations_with_IBM.read_data import reader


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