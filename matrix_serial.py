import numpy as np
import time

# Create random matrices
N = 500
A = np.random.rand(N, N)
B = np.random.rand(N, N)

start_time = time.time()
C = np.dot(A, B)
end_time = time.time()

print("Serial Execution Time:", end_time - start_time)
