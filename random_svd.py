import math
import time

import numpy as np
import pandas as pd
from mpi4py import MPI

# A = pd.read_csv('data/baseball_features.csv').to_numpy()

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

input = None
if rank == 0:
    input = np.random.normal(size=(1000, 1000))

A = comm.scatter([input for _ in range(num_procs)])

k = 10
def points_per_rank(num_points):
    return int((num_points + num_procs - 1) / num_procs)

def dot(v1, v2):
    assert len(v1) == len(v2)
    return sum(v1[i] * v2[i] for i in range(len(v1)))

cols_per_rank = points_per_rank(k)
row_start_index = rank * cols_per_rank
row_end_index = min((rank + 1) * cols_per_rank, k)

t0 = time.time()

G_cols = np.random.normal(size=(A.shape[1], row_end_index - row_start_index))
piece = np.zeros((A.shape[0], G_cols.shape[1]))
for j in range(G_cols.shape[1]):
    col = G_cols[:,j]
    for i in range(A.shape[0]):
        row = A[i,:]
        piece[i,j] = sum(row[l] * col[l] for l in range(A.shape[0]))

print(time.time() - t0)
pieces = comm.gather(piece)

Q = None
if rank == 0:
    AG = np.column_stack(pieces)
    Q = np.zeros(AG.shape)
    print(time.time() - t0)
    print(AG.shape)

    for i in range(Q.shape[1]):
        q_i = AG[:, i]
        for j in range(0, i):
            q_j = Q[:, j]
            q_i = q_i - dot(q_i, q_j) * q_j
        Q[:, i] = q_i / math.sqrt(sum(q_i_elmt**2 for q_i_elmt in q_i))

    # print([np.linalg.norm(Q[:, i]) for i in range(Q.shape[1])])

Q = comm.scatter([Q for _ in range(num_procs)])
