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
dim = 10
if rank == 0:
    # input = np.random.randint(-10, high=10, size=(dim, dim))
    input = pd.read_csv('data/baseball_features.csv').to_numpy()[:1000,]

A = linear_algebra_funcs.scatter_matrix(input)

k = int(A.shape[0] / 20)

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

print(f'Calculated Q_TA in {time.time() - t0}s')

# Run standard SVD algorithm
t0 = time.time()
Q_TA_reduced = Q_TA.copy()
u_sigma_v = []
sigma, u, v = None, None, None
for i in range(k):
    if rank == 0:
        print(f'Starting singular vectors/value {i}')
    if rank == 0 and u is not None:
        outer_prod = np.outer(u, v)
        Q_TA_reduced -= sigma * outer_prod

    sigma, u, v = None, None, None
    if Q_TA.shape[0] > Q_TA.shape[1]:
        v = linear_algebra_funcs.parallel_power_method(Q_TA_reduced)
        u_unnormalized = linear_algebra_funcs.parallel_matmul(Q_TA, v, matrix_vector=True)
        if rank == 0:
            sigma = sum(elmt**2 for elmt in u_unnormalized)
            u = u_unnormalized / sigma
    else:
        transposed = Q_TA_reduced.T
        u = linear_algebra_funcs.parallel_power_method(transposed)  # next singular vector
        v_unnormalized = linear_algebra_funcs.parallel_matmul(transposed, u, matrix_vector=True)
        if rank == 0:
            sigma = sum(elmt**2 for elmt in v_unnormalized)
            v = v_unnormalized / sigma

    if rank == 0:
        u_sigma_v.append((u, sigma, v))

if rank == 0:
    print(f'Standard SVD algo took: {time.time() - t0}')
    U, Sigma, V = [np.array(x) for x in zip(*u_sigma_v)]
    U = U.reshape(U.shape[:2])
    V = V.reshape(V.shape[:2])
    print(U)
    print(Sigma)
    print(V)
    print(f'Q_TA: {Q_TA.shape}')
    print(f'U: {U.shape}')
    print(f'V: {V.shape}')
