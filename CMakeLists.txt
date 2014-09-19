CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)
PROJECT(mfg)

#SET(CMAKE_INCLUDE_CURRENT_DIR ON)

# Set search directory for looking for our custom CMake scripts to
# look for SuiteSparse, QGLViewer, and Eigen3.
LIST(APPEND CMAKE_MODULE_PATH ${mfg_SOURCE_DIR}/cmake_modules)


################################################################################
# Set the version of the kernel project
################################################################################
set(mfg_MAJOR_VERSION 0)
set(mfg_MINOR_VERSION 1)
set(mfg_PATCH_VERSION 0)
set(mfg_VERSION ${mfg_MAJOR_VERSION}.${mfg_MINOR_VERSION}.${mfg_PATCH_VERSION})


################################################################################
# Set the output directory for the build executables and libraries
################################################################################
SET(mfg_RUNTIME_OUTPUT_DIRECTORY ${mfg_SOURCE_DIR}/bin CACHE PATH "Target for the binaries")
IF(WIN32)
   SET(mfg_LIBRARY_OUTPUT_DIRECTORY ${mfg_SOURCE_DIR}/bin CACHE PATH "Target for the libraries")
ELSE(WIN32)
   SET(mfg_LIBRARY_OUTPUT_DIRECTORY ${mfg_SOURCE_DIR}/lib CACHE PATH "Target for the libraries")
ENDIF(WIN32)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${mfg_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${mfg_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${mfg_RUNTIME_OUTPUT_DIRECTORY})


################################################################################
# Detect OS and define macros appropriately
################################################################################
IF(WIN32)
   ADD_DEFINITIONS(-DWINDOWS)
   MESSAGE(STATUS "Compiling on Windows")
   IF(MSVC)
      MESSAGE(STATUS "Compiling with MSVC")
      ADD_DEFINITIONS(-DNOMINMAX)
      ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
      # SSE2 optimizations
      ADD_DEFINITIONS("/arch:SSE2")
      IF (BUILD_SHARED_LIBS)
         # disable warning on missing DLL interfaces
         ADD_DEFINITIONS("/wd4251")
      ENDIF()
   ENDIF(MSVC)
ELSEIF(CYGWIN)
   MESSAGE(STATUS "Compiling on Cygwin")
   ADD_DEFINITIONS(-DCYGWIN)
ELSEIF(APPLE)
   ADD_DEFINITIONS(-DUNIX)
   MESSAGE(STATUS "Compiling on OSX")
ELSEIF(UNIX)
   ADD_DEFINITIONS(-DUNIX)
   MESSAGE(STATUS "Compiling on Unix")
ENDIF(WIN32)


################################################################################
# C++11 support
################################################################################
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
IF(COMPILER_SUPPORTS_CXX11)
   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ELSEIF(COMPILER_SUPPORTS_CXX0X)
   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
   MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()


################################################################################
# Project Dependencies:
# TODO: make sure all req packages are found
################################################################################
# Qt5
FIND_PACKAGE(Qt5Core REQUIRED)
FIND_PACKAGE(Qt5OpenGL REQUIRED)
SET(CMAKE_AUTOMOC ON)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(CSparse REQUIRED)
FIND_PACKAGE(G2O REQUIRED)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
#FIND_PACKAGE( REQUIRED)
#FIND_PACKAGE( REQUIRED)


################################################################################
# Variables
################################################################################
SET(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
SET(HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/includes)
SET(EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext)
SET(LSD_HEADERS ${HEADER_DIR}/lsd/lsd.h)
SET(LM_HEADER_DIR ${HEADER_DIR}/levmar)
SET(LM_HEADERS 
   ${LM_HEADER_DIR}/compiler.h
   ${LM_HEADER_DIR}/levmar.h
   ${LM_HEADER_DIR}/lm.h
   ${LM_HEADER_DIR}/misc.h
)

LINK_DIRECTORIES(${LIB_DIR})
INCLUDE_DIRECTORIES(
   ${HEADER_DIR}
   ${EXTERN_DIR}
   ${Qt5Gui_INCLUDE_DIR}
   ${EIGEN3_INCLUDE_DIR}
   ${CSPARSE_INCLUDE_DIR}
)


#message(STATUS "csparse_include_dir = ${CSPARSE_INCLUDE_DIR}")
#message(STATUS "qt_include_dir***** = ${Qt5OpenGL_INCLUDE_DIR}")
message(STATUS "qt_**************** = ${Qt5OpenGL_LIBRARIES}")
message(STATUS "OPENGL_************ = ${OPENGL_LIBRARIES}")
message(STATUS "GLUT_************** = ${GLUT_LIBRARIES}")
message(STATUS "CSPARSE_*********** = ${CSPARSE_LIBRARY}")
message(STATUS "G2O_CORE_********** = ${G2O_CORE_LIBRARY}")
message(STATUS "G2O_STUFF_********* = ${G2O_STUFF_LIBRARY}")
message(STATUS "G2O_SLAM3D_******** = ${G2O_TYPES_SLAM3D}")
message(STATUS "G2O_SBA_*********** = ${G2O_TYPES_SBA}")
message(STATUS "G2O_CSPARSE_******* = ${G2O_SOLVER_CSPARSE}")
message(STATUS "G2O_CSPARSE_EXT_*** = ${G2O_SOLVER_CSPARSE_EXTENSION}")
message(STATUS "BLAS_************** = ${BLAS_LIBRARIES}")
message(STATUS "LAPACK_************ = ${LAPACK_LIBRARIES}")


# Make dependencies first
ADD_SUBDIRECTORY(ext/levmar-2.6)
ADD_SUBDIRECTORY(ext/lsd-1.5)

# Make this project
ADD_SUBDIRECTORY(src)

