# CUDA language support
cmake_minimum_required(VERSION 3.8)

project(CUDA-MatMul LANGUAGES CXX CUDA)

# given matrix multiplication methods
add_executable(gpu_matmul_1 gpu_matmul_1.cu)
add_executable(gpu_matmul_2 gpu_matmul_2.cu)
add_executable(gpu_matmul_3 gpu_matmul_3.cu)