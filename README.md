# Distributed Matrix Multiplication using MPI

##Objective
To implement and evaluate the performance of matrix multiplication across multiple nodes using MPI (Message Passing Interface).

## Project Description
This project compares serial and MPI-based distributed matrix multiplication in terms of execution time and scalability using `mpi4py`. We split the matrix among processes, perform local computation, and gather the result at the root process.

## Requirements

- Python 3.x
- numpy
- mpi4py
- MPI environment (e.g., MPICH or OpenMPI)

Install dependencies:
```bash
sudo apt install mpich
pip install mpi4py numpy
