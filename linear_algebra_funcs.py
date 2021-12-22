import math
import time

from mpi4py import MPI
import numpy as np

import util


def scatter_matrix(send_buf):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    matrix_shape = (None, None)
    if rank == 0:
        matrix_shape = send_buf.shape
    matrix_shape = comm.bcast(matrix_shape)
    rows = []
    for i in range(matrix_shape[0]):
        if rank == 0:
            row = send_buf[i,].copy()
        else:
            row = np.empty(matrix_shape[1])
        comm.Bcast([row, MPI.FLOAT])
        rows.append(row)
    return np.row_stack(rows)


def scatter_vector(send_buf):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    vector_shape = 0
    if rank == 0:
        vector_shape = send_buf.shape[0]
    vector_shape = comm.bcast(vector_shape)
    if rank == 0:
        vec = send_buf.copy()
    else:
        vec = np.empty(vector_shape)
    comm.Bcast([vec, MPI.FLOAT])
    return vec


def qr_factorization(A):
    Q = np.empty(A.shape)

    for i in range(Q.shape[1]):
        q_i = A[:, i]
        for j in range(0, i):
            q_j = Q[:, j]
            q_i = q_i - util.dot(q_i, q_j) * q_j
        Q[:, i] = q_i / math.sqrt(sum(q_i_elmt**2 for q_i_elmt in q_i))

    return Q


def parallel_matmul(A, B, matrix_vector):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    if matrix_vector:
        B = scatter_vector(B)
    else:
        B = scatter_matrix(B)

    t0 = time.time()
    if rank == 0:
        for i in range(1, num_procs):
            row_start_index, row_end_index = util.start_end_index(A.shape[0], num_procs, i)
            # print(f'Sending rows {row_start_index} - {row_end_index} to rank {i}')
            comm.isend(row_end_index - row_start_index, dest=i, tag=i)
            for row_index in range(row_start_index, row_end_index):
                row_to_send = A[row_index,].copy()
                comm.Isend([row_to_send, MPI.FLOAT], dest=i, tag=i)

        row_start_index, row_end_index = util.start_end_index(A.shape[0], num_procs, 0)
        A_piece = A[row_start_index:row_end_index,:]
    else: # Put together piece of A to calculate a piece of AB with
        rows = []
        rows_to_recv = comm.recv(source=0, tag=rank)
        A_piece = np.empty((rows_to_recv, B.shape[0]))
        for i in range(rows_to_recv):
            comm.Recv([A_piece[i,], MPI.FLOAT], source=0, tag=rank)

    # print(f'rank {rank} A_piece construction {A_piece.shape}x{B.shape}: {time.time() - t0},  matrix_vector={matrix_vector}')
    # Calculate a piece of QT_A
    num_cols = 1 if matrix_vector else B.shape[1]
    piece = np.empty((A_piece.shape[0], num_cols))
    for i in range(A_piece.shape[0]):
        for j in range(num_cols):
            if matrix_vector:
                piece[i] = util.dot(A_piece[i,], B)
            else:
                piece[i, j] = util.dot(A_piece[i,], B[:,j])

    pieces = comm.gather(piece)
    if rank == 0:
        return np.concatenate(pieces)


def parallel_power_method(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    B_send_buf = None
    mm_arg1, mm_arg2 = None, None
    if rank == 0 and A.shape[0] > A.shape[1]:
        mm_arg1 = A.transpose()
        mm_arg2 = A
    elif rank == 0:
        mm_arg1 = A
        mm_arg2 = A.transpose()

    B_send_buf = parallel_matmul(mm_arg1, mm_arg2, matrix_vector=False)
    B = scatter_matrix(B_send_buf)
    v_send_buf = None
    if rank == 0:
        v_send_buf = util.normalize(np.random.normal(size=min(*A.shape)))
    v = scatter_vector(v_send_buf)

    prev = None
    iterations = 0
    converged = False
    while True:
        iterations += 1
        prev = v
        if rank == 0:
            pass

        v_send_buf = parallel_matmul(B, prev, matrix_vector=True)
        if rank == 0:
            v_send_buf = util.normalize(v_send_buf)
        v = scatter_vector(v_send_buf)
        if abs(util.dot(v, prev)) > 1 - epsilon:
            if rank == 0:
                print(f"converged in {iterations} iterations!")
            return v
