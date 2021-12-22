import argparse
import math
import time

import numpy as np
import pandas as pd
from mpi4py import MPI

import linear_algebra_funcs
import util


comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

input = None
if rank == 0:
    input = pd.read_csv('data/baseball_features.csv').to_numpy()[:1000,]

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--randomized', action='store_true', default=False)
args = parser.parse_args()
randomized = args.randomized
A = linear_algebra_funcs.scatter_matrix(input)
k = int(A.shape[0] / 20)
Q_TA = None
if randomized:

    row_start_index, row_end_index = util.start_end_index(k, num_procs, rank)
    print(f'rank {rank} will handle rows {row_start_index} to {row_end_index}')

    t0 = time.time()

    # Calculate a piece of AG - no need to scatter G, as all entries are random
    G_cols = np.random.normal(size=(A.shape[1], row_end_index - row_start_index))
    piece = np.empty((A.shape[0], G_cols.shape[1]))
    for j in range(G_cols.shape[1]):
        for i in range(A.shape[0]):
            piece[i,j] = util.dot(A[i,], G_cols[:,j])

    print(f'rank {rank} A * G ({A.shape} x {G_cols.shape}): {time.time() - t0}')
    pieces = comm.gather(piece)

    t0 = time.time()
    Q_send_buf = None
    if rank == 0:
        AG = np.column_stack(pieces)
        # Calculate Q, which is not parallelizable
        Q_send_buf = linear_algebra_funcs.qr_factorization(AG)

    Q = linear_algebra_funcs.scatter_matrix(Q_send_buf)

    t0 = time.time()
    Q_TA_send_buf = linear_algebra_funcs.parallel_matmul(Q.transpose(), A, matrix_vector=False)
    Q_TA = linear_algebra_funcs.scatter_matrix(Q_TA_send_buf)

    if rank == 0:
        print(f'Calculated Q_TA in {time.time() - t0}s')

# Run standard SVD algorithm
t0 = time.time()
SVD_matrix = Q_TA.copy() if randomized else A.copy()
SVD_matrix_reduced = SVD_matrix.copy()
u_sigma_v = []
sigma, u, v = None, None, None
for i in range(k):
    if rank == 0:
        print(f'Starting singular vectors/value {i}')
    if rank == 0 and u is not None:
        outer_prod = np.outer(u, v)
        SVD_matrix_reduced -= sigma * outer_prod

    sigma, u, v = None, None, None
    if SVD_matrix.shape[0] > SVD_matrix.shape[1]:
        v = linear_algebra_funcs.parallel_power_method(SVD_matrix_reduced)
        u_unnormalized = linear_algebra_funcs.parallel_matmul(SVD_matrix, v, matrix_vector=True)
        if rank == 0:
            sigma = math.sqrt(sum(elmt**2 for elmt in u_unnormalized))
            u = u_unnormalized / sigma
    else:
        transposed = SVD_matrix_reduced.T
        u = linear_algebra_funcs.parallel_power_method(transposed)  # next singular vector
        v_unnormalized = linear_algebra_funcs.parallel_matmul(transposed, u, matrix_vector=True)
        if rank == 0:
            sigma = math.sqrt(sum(elmt**2 for elmt in v_unnormalized))
            v = v_unnormalized / sigma

    if rank == 0:
        u_sigma_v.append((u, sigma, v))

U_tilde = None
if rank == 0:
    print(f'Standard SVD algo took: {time.time() - t0}')
    U, Sigma, V = [np.array(x) for x in zip(*u_sigma_v)]
    U_tilde = U.reshape(U.shape[:2]).transpose()
    V = V.reshape(V.shape[:2]).transpose()

if randomized:
    U = linear_algebra_funcs.parallel_matmul(Q, U_tilde, matrix_vector=False)
elif rank == 0:
    U = U_tilde

if rank == 0:
    print(U)
    print(Sigma)
    print(V)
    print(f'A: {A.shape}')
    if randomized:
        print(f'reduced A: {Q_TA.shape}')
    print(f'U: {U.shape}')
    print(f'Sigma: {Sigma.shape}')
    print(f'V: {V.shape}')
