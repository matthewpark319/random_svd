import math
import time

import numpy as np
import pandas as pd
from mpi4py import MPI

import util

# A = pd.read_csv('data/baseball_features.csv').to_numpy()

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

input = None
if rank == 0:
    input = np.random.normal(size=(100, 100))

A = comm.scatter([input for _ in range(num_procs)])

k = 10

row_start_index, row_end_index = util.start_end_index(k, num_procs, rank)
print(f'rank {rank} will handle rows {row_start_index} to {row_end_index}')

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

    for i in range(Q.shape[1]):
        q_i = AG[:, i]
        for j in range(0, i):
            q_j = Q[:, j]
            q_i = q_i - util.dot(q_i, q_j) * q_j
        Q[:, i] = q_i / math.sqrt(sum(q_i_elmt**2 for q_i_elmt in q_i))

    Q_T = Q.transpose()
    for i in range(1, num_procs):
        row_start_index, row_end_index = util.start_end_index(Q_T.shape[0], num_procs, i)
        comm.isend(row_end_index - row_start_index, dest=i, tag=i)
        for row_index in range(row_start_index, row_end_index):
            comm.isend(Q_T[row_index,], dest=i, tag=i)

if rank > 0:
    rows = []
    rows_to_recv = comm.recv(source=0, tag=rank)
    for _ in range(rows_to_recv):
        row = comm.recv(source=0, tag=rank)
        rows.append(row)
    Q_T_piece = np.vstack(rows)
    piece = Q_T_piece @ A
elif rank == 0:
    row_start_index, row_end_index = util.start_end_index(Q_T.shape[0], num_procs, 0)
    piece = Q_T[row_start_index:row_end_index,:] @ A

pieces = comm.gather(piece)

if rank == 0:
    Q_T_A = np.concatenate(pieces)
    print(Q_T_A.shape)
    print(Q.shape)
