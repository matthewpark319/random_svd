# Randomized SVD in parallel

Running this code requires python 3.8.10 with installations of numpy 1.19.5 and mpi4py 3.1.3. On the author's machine, these were installed using pip version 20.0.2:
```
pip install numpy
pip install mpi4py
```

To run the SVD implementation,
```
mpiexec -np <number_of_processes> python3 randomized_svd.py
```

To run the randomized SVD implementation, run this command:
```
mpiexec -np <number_of_processes> python3 randomized_svd.py --randomized
```
