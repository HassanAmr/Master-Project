cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Debug)
project( ThesisProject )
find_package( OpenCV )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( run src/main.cpp )
target_link_libraries( run ${OpenCV_LIBS} )
