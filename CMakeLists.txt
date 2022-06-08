cmake_minimum_required(VERSION 3.20)
project(GameOfLifeCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(GameOfLifeCuda streams.cu)

#set_target_properties(GameOfLifeCuda PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
