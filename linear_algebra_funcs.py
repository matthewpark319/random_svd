import math

import numpy as np

import util

def qr_factorization(A):
    Q = np.empty(A.shape)

    for i in range(Q.shape[1]):
        q_i = A[:, i]
        for j in range(0, i):
            q_j = Q[:, j]
            q_i = q_i - util.dot(q_i, q_j) * q_j
        Q[:, i] = q_i / math.sqrt(sum(q_i_elmt**2 for q_i_elmt in q_i))

    return Q
