import time

import pandas as pd
import numpy as np
from mpi4py import MPI

A = pd.read_csv('data/baseball_features.csv').to_numpy()
B = pd.read_csv('data/coefs.csv').to_numpy()

comm = MPI.COMM_WORLD
t0 = time.time()

num_procs = comm.Get_size()
rank = comm.Get_rank()

num_rows = A.shape[0]
rows_per_rank = int((num_rows + num_procs - 1) / num_procs)
row_start_index = rank * rows_per_rank
row_end_index = min((rank + 1) * rows_per_rank, num_rows)
print(f'rank {rank} will handle rows {row_start_index} to {row_end_index}')

piece = A[row_start_index:row_end_index,] @ B
pieces = comm.gather(piece)
if rank == 0:
    result = np.concatenate(pieces)
    print(result)
    print(result.shape)

    print(time.time() - t0)
