cmake_minimum_required(VERSION 2.8.11)
project(mfg)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Project Dependencies:
# Qt5
set(CMAKE_AUTOMOC ON)
find_package(Qt5Core REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)

# Variables
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/includes)
set(LSD_HEADERS ${HEADER_DIR}/lsd/lsd.h)
set(LM_HEADER_DIR ${HEADER_DIR}/levmar-2.6)
set(LM_HEADERS 
   ${LM_HEADER_DIR}/compiler.h
   ${LM_HEADER_DIR}/levmar.h
   ${LM_HEADER_DIR}/lm.h
   ${LM_HEADER_DIR}/misc.h
)

link_directories(${LIB_DIR})
include_directories(${HEADER_DIR})

add_subdirectory(src)

