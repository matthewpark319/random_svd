import time

import numpy as np
import pandas as pd
from mpi4py import MPI

# A = pd.read_csv('data/baseball_features.csv').to_numpy()
A = np.random.normal(size=(10000, 10000))

k = 1000

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

def points_per_rank(num_points):
    return int((num_points + num_procs - 1) / num_procs)

cols_per_rank = points_per_rank(k)
row_start_index = rank * cols_per_rank
row_end_index = min((rank + 1) * cols_per_rank, k)

t0 = time.time()

piece = A @ np.random.normal(size=(A.shape[1], row_end_index - row_start_index))
print(time.time() - t0)
pieces = comm.gather(piece)


if rank == 0:
    AG = np.column_stack(pieces)
    print(time.time() - t0)
    print(AG.shape)
    # Q, _ = np.linalg.qr(AG)

