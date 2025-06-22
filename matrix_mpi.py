from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 500
rows_per_process = N // size

A = None
B = None
if rank == 0:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)

# Scatter rows of A to all processes
local_A = np.zeros((rows_per_process, N))
comm.Scatter(A, local_A, root=0)

# Broadcast B to all processes
if rank != 0:
    B = np.empty((N, N))
comm.Bcast(B, root=0)

# Local multiplication
start_time = time.time()
local_C = np.dot(local_A, B)
end_time = time.time()

# Gather results
C = None
if rank == 0:
    C = np.zeros((N, N))
comm.Gather(local_C, C, root=0)

# Display time on rank 0
if rank == 0:
    print("Parallel Execution Time:", end_time - start_time)
