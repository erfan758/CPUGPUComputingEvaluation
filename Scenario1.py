
import numpy as np
import time
import torch

# Define the matrix sizes
matrix_size = 1000
matrix_a = np.random.rand(matrix_size, matrix_size)
matrix_b = np.random.rand(matrix_size, matrix_size)

# CPU Matrix Multiplication
def cpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    # Perform matrix multiplication using CPU
    cpu_result = np.dot(matrix_a, matrix_b)

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time

# GPU Matrix Multiplication
def gpu_matrix_multiplication(matrix_a, matrix_b):
    start_time = time.time()

    # Perform matrix multiplication using GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_a = torch.FloatTensor(matrix_a).to(device)
    tensor_b = torch.FloatTensor(matrix_b).to(device)
    tensor_result = torch.mm(tensor_a, tensor_b)
    cpu_result = tensor_result.cpu().numpy()

    end_time = time.time()
    execution_time = end_time - start_time
    return cpu_result, execution_time

# Run CPU Matrix Multiplication
cpu_result, cpu_execution_time = cpu_matrix_multiplication(matrix_a, matrix_b)

# Run GPU Matrix Multiplication
gpu_result, gpu_execution_time = gpu_matrix_multiplication(matrix_a, matrix_b)

# Compare the results (optional)
print("CPU Result:")
print(cpu_result)
print("GPU Result:")
print(gpu_result)

# Print the execution times
print("CPU Execution Time:", cpu_execution_time, "seconds")
print("GPU Execution Time:", gpu_execution_time, "seconds")

