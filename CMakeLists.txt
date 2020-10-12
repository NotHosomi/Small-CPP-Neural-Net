cmake_minimum_required(VERSION 3.12)
project(MyNN)

set(CMAKE_CXX_STANDARD 14)

add_executable(MyNN main.cpp Net.cpp Net.hpp Neuron.cpp Neuron.hpp)