# CUDA language support
cmake_minimum_required(VERSION 3.8)
# C++ and CUDA
project(adjacent_difference LANGUAGES CXX CUDA)

# add exe
add_executable(${PROJECT_NAME} ad.cu)