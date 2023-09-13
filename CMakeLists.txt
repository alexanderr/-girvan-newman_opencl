cmake_minimum_required(VERSION 3.19)
project(opencl_graph)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)

add_executable(opencl_graph main.cpp graph_kernel.h open_cl_util.cpp open_cl_util.h)
target_link_libraries(opencl_graph OpenCL::OpenCL)
configure_file(dijkstra.cl ${CMAKE_CURRENT_BINARY_DIR}/dijkstra.cl COPYONLY)
configure_file(edge_betweenness.cl ${CMAKE_CURRENT_BINARY_DIR}/edge_betweenness.cl COPYONLY)
