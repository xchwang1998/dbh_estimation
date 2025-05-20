cmake_minimum_required(VERSION 3.0.2)
project(DBH_Estimation)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find PCL, version 1.8 (system), may confilict with GTSAM or smallGICP 
# find_package(PCL 1.8 REQUIRED QUIET)

# find PCL, self builded, for smallGICP
set(PCL_DIR "/home/xiaochen/pcl-1.12.0/share/pcl-1.12")
find_package(PCL 1.12 REQUIRED)
# message(${PCL_INCLUDE_DIRS})
# message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")

# find third party library
find_package(OpenCV REQUIRED QUIET)
find_package(Eigen3 REQUIRED)
find_package(libLAS REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

# link directories
link_directories(
	include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
)

# set the global path variable
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR})
add_compile_definitions(PROJECT_PATH="${PROJECT_ROOT}")

# find file's global path
file(GLOB utils_sources ./src/utils/*.cpp)
message(${utils_sources})
##########################################################
########################Demo Node#########################
##########################################################
# dbh estimation
add_executable(dbh_estimation ./src/demo/main.cpp ${utils_sources})
target_compile_options(dbh_estimation PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(dbh_estimation
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  ${libLAS_LIBRARIES}
  libCSF.a
)

##########################################################
########################Demo Node#########################
##########################################################
