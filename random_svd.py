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
dim = 1000
if rank == 0:
    input = np.random.normal(size=(dim, dim))

A = comm.scatter([input for _ in range(num_procs)])

k = 10

row_start_index, row_end_index = util.start_end_index(k, num_procs, rank)
print(f'rank {rank} will handle rows {row_start_index} to {row_end_index}')

t0 = time.time()

G_cols = np.random.normal(size=(A.shape[1], row_end_index - row_start_index))
piece = np.empty((A.shape[0], G_cols.shape[1]))
for j in range(G_cols.shape[1]):
    for i in range(A.shape[0]):
        piece[i,j] = util.dot(A[i,], G_cols[:,j])

print(f'rank {rank} A * G ({A.shape} x {G_cols.shape}): {time.time() - t0}')
pieces = comm.gather(piece)

Q = None
if rank == 0:
    AG = np.column_stack(pieces)
    Q = linear_algebra_funcs.qr_factorization(AG)
    Q_T = Q.transpose()
    for i in range(1, num_procs):
        row_start_index, row_end_index = util.start_end_index(Q_T.shape[0], num_procs, i)
        comm.isend(row_end_index - row_start_index, dest=i, tag=i)
        for row_index in range(row_start_index, row_end_index):
            row_to_send = Q_T[row_index,].copy()
            comm.Isend([row_to_send, MPI.FLOAT], dest=i, tag=i)

t0 = time.time()
if rank == 0:
    row_start_index, row_end_index = util.start_end_index(Q_T.shape[0], num_procs, 0)
    Q_T_piece = Q_T[row_start_index:row_end_index,:]
elif rank > 0:
    rows = []
    rows_to_recv = comm.recv(source=0, tag=rank)
    Q_T_piece = np.empty((rows_to_recv, A.shape[0]))
    for i in range(rows_to_recv):
        Q_T_piece[i,] = comm.Recv([Q_T_piece[i,], MPI.FLOAT], source=0, tag=rank)

print(f'rank {rank} Q_T_piece construction ({Q_T_piece.shape}): {time.time() - t0}')

piece = np.empty((Q_T_piece.shape[0], A.shape[1]))
for i in range(Q_T_piece.shape[0]):
    for j in range(A.shape[1]):
        piece[i, j] = util.dot(Q_T_piece[i,], A[:,j])

print(f'rank {rank} Q_T * A ({Q_T_piece.shape} x {A.shape}): {time.time() - t0}')
pieces = comm.gather(piece)

if rank == 0:
    Q_T_A = np.concatenate(pieces)

    # TODO: add SVD(Q_T_A) here
