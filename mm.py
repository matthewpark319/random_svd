from mpi4py import MPI
import pandas as pd
import numpy as np

features = pd.read_csv('data/baseball_features.csv').to_numpy()
coefs = pd.read_csv('data/coefs.csv').to_numpy()


comm = MPI.COMM_WORLD

def multiply(A, B):
    num_procs = comm.Get_size()
    rank = comm.Get_rank()

    num_rows = A.shape[0]
    rows_per_rank = int((num_rows + num_procs - 1) / num_procs)
    row_start_index = rank * rows_per_rank
    row_end_index = min((rank + 1) * rows_per_rank, num_rows)
    print(f'rank {rank} will handle rows {row_start_index} to {row_end_index}')

    piece = A[row_start_index:row_end_index,] @ B
    result = np.empty((A.shape[0], B.shape[1]))
    if rank != 0:
        comm.isend(piece, 0, rank)
    else:
        result[:row_end_index,] = piece
        for i in range(1, num_procs):
            next_piece = comm.recv(source=i, tag=i)
            result[i * rows_per_rank:min((i + 1) * rows_per_rank, num_rows)] = next_piece
            print(f'inserted data from rank {i}')
    return result

# if comm.Get_rank() == 0:
product = multiply(features, coefs)
print(product)
