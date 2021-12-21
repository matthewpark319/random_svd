import math
import time

import numpy as np
import pandas as pd
from mpi4py import MPI

import linear_algebra_funcs
import util

# A = pd.read_csv('data/baseball_features.csv').to_numpy()

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

input = None
dim = 100
if rank == 0:
    input = np.random.normal(size=(dim, dim))

A = comm.scatter([input for _ in range(num_procs)])

k = 10

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
Q = None
if rank == 0:
    AG = np.column_stack(pieces)

    # Calculate Q, which is not parallelizable
    Q = linear_algebra_funcs.qr_factorization(AG)

Q = comm.scatter([Q for _ in range(num_procs)])

t0 = time.time()
Q_TA = linear_algebra_funcs.parallel_matmul(Q.transpose(), A)
Q_TA = comm.scatter([Q_TA for _ in range(num_procs)])
print(f'Calculated Q_TA in {time.time() - t0}s')

t0 = time.time()
if rank == 0:
    # TODO: add SVD(Q_TA) here
    pass
