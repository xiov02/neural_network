cmake_minimum_required(VERSION 3.10)
project(MyNeuronProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add sources
add_executable(main
    src/main.cpp
    src/neuron/neuron.cpp
    src/neuron/inputNeuron.cpp
    src/layer/hiddenLayer.cpp
    src/layer/inputLayer.cpp
    src/network/multilayerPerceptron.cpp
)

# Compilation options
add_compile_options(-Wall -Wextra -O2)

# directories for includes 
target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
