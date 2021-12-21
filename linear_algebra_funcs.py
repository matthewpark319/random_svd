import math
import time

from mpi4py import MPI
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


def parallel_matmul(A, B):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    B = comm.scatter([B for _ in range(num_procs)])

    t0 = time.time()
    if rank == 0:
        for i in range(1, num_procs):
            row_start_index, row_end_index = util.start_end_index(A.shape[0], num_procs, i)
            print(f'Sending rows {row_start_index} - {row_end_index} to rank {i}')
            comm.isend(row_end_index - row_start_index, dest=i, tag=i)
            for row_index in range(row_start_index, row_end_index):
                row_to_send = A[row_index,].copy()
                comm.Isend([row_to_send, MPI.FLOAT], dest=i, tag=i)

        row_start_index, row_end_index = util.start_end_index(A.shape[0], num_procs, 0)
        A_piece = A[row_start_index:row_end_index,:]
    else: # Calculate which piece of A to calculate a piece of AA with
        rows = []
        rows_to_recv = comm.recv(source=0, tag=rank)
        A_piece = np.empty((rows_to_recv, B.shape[0]))
        for i in range(rows_to_recv):
            comm.Recv([A_piece[i,], MPI.FLOAT], source=0, tag=rank)

    # Calculate a piece of QT_A
    num_cols = B.shape[1] if len(B.shape) > 1 else 1
    piece = np.empty((A_piece.shape[0], num_cols))
    for i in range(A_piece.shape[0]):
        for j in range(num_cols):
            if num_cols > 1:
                piece[i, j] = util.dot(A_piece[i,], B[:,j])
            else:
                piece[i] = util.dot(A_piece[i,], B)
            # piece[i, j] = A_piece[i,].dot(B[:,j])

    print(f'rank {rank} A_piece construction ({A_piece.shape}): {time.time() - t0}')

    pieces = comm.gather(piece)
    if rank == 0:
        concat = np.concatenate(pieces)
        if num_cols > 1:
            return concat
        return concat[:,0]


def parallel_power_method(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    A = comm.scatter([A for _ in range(num_procs)])
    if A.shape[0] > A.shape[1]:
        B = parallel_matmul(A.transpose(), A)
    else:
        B = parallel_matmul(A, A.transpose())
    B = comm.scatter([B for _ in range(num_procs)])

    v = None
    if rank == 0:
        v = util.normalize(np.random.normal(size=min(*A.shape)))

    prev = None
    iterations = 0
    while True:
        if rank == 0:
            iterations += 1
            prev = v

        v = parallel_matmul(B, prev)
        if rank == 0:
            v = util.normalize(v)
            if abs(util.dot(v, prev)) > 1 - epsilon:
                print("converged in {} iterations!".format(iterations))
                return v
