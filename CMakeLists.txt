CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)
PROJECT(mfg)

# Set search directory for looking for our custom CMake scripts to
# look for SuiteSparse, QGLViewer, and Eigen3.
LIST(APPEND CMAKE_MODULE_PATH ${mfg_SOURCE_DIR}/cmake_modules)

# for CMake 3.0 and above, this will suppress warnings about old policies
IF(POLICY CMP0043)
CMAKE_POLICY(SET CMP0043 OLD)
ENDIF(POLICY CMP0043)


################################################################################
# Set the version of the kernel project
################################################################################
set(mfg_MAJOR_VERSION 0)
set(mfg_MINOR_VERSION 5)
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
   #SET(CC "${CC} /opt/local/bin")
   #SET(CXX "${CXX} /opt/local/bin")
   #SET(CMAKE_C_COMPILER gcc-mp-4.9)
   #SET(CMAKE_CXX_COMPILER g++-mp-4.9)
   #SET(CMAKE_C_COMPILER /usr/local/bin/gcc-mp-4.9)
   #SET(CMAKE_CXX_COMPILER /usr/local/bin/g++-mp-4.9)
   MESSAGE(STATUS "Compiling on OSX")
ELSEIF(UNIX)
   ADD_DEFINITIONS(-DUNIX)
   MESSAGE(STATUS "Compiling on Unix")
ENDIF(WIN32)


################################################################################
# C++11 support
#
# Notes:
# OS X 10.8 Mountain Lion and above default to using the clang compiler,
# which does not use C++ 11 unless explicitly told to, as below.
################################################################################
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
IF(COMPILER_SUPPORTS_CXX11)
   IF(APPLE)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++ -O2")
   ELSE(APPLE)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
   ENDIF(APPLE)
   MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} supports C++11")
ELSEIF(COMPILER_SUPPORTS_CXX0X)
   IF(APPLE)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -stdlib=libc++ -O2")
   ELSE(APPLE)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O2")
   ENDIF(APPLE)
   MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} supports C++0X")
ELSE()
   MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()


################################################################################
# Add a couple custom targets to quickly make a debug or release version
################################################################################
SET(CMAKE_BUILD_TYPE Release)

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)


################################################################################
# Project Dependencies:
################################################################################
# Qt5 via Homebrew on OS X requires some fancy PREFIX_PATH changes
IF(APPLE)
   SET(CMAKE_PREFIX_PATH /usr/local/opt/qt5)
ENDIF(APPLE)
FIND_PACKAGE(Qt5Core REQUIRED)
FIND_PACKAGE(Qt5OpenGL REQUIRED)
FIND_PACKAGE(Qt5Xml REQUIRED) # required for libQGLViewer

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(CSparse REQUIRED)
FIND_PACKAGE(G2O REQUIRED)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
add_definitions(-DUSE_GROUND_PLANE)#use ground plane

################################################################################
# Variables
################################################################################
SET(LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
SET(HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/includes)
SET(EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext)

# LSD project includes
SET(LSD_HEADER_DIR ${HEADER_DIR}/lsd)
SET(LSD_HEADERS ${LSD_HEADER_DIR}/lsd.h)

# LevMar (LM) project includes
SET(LM_HEADER_DIR ${HEADER_DIR}/levmar)
SET(LM_HEADERS
   ${LM_HEADER_DIR}/compiler.h
   ${LM_HEADER_DIR}/levmar.h
   ${LM_HEADER_DIR}/lm.h
   ${LM_HEADER_DIR}/misc.h
)

# We want to skip the extra apps and examples to speed the build up for G2O.
IF(G2O_NOT_FOUND)
   MESSAGE(STATUS "G2O was not found, will build a local version")

   # G2O, BLAS, LAPACK headers and libraries
   SET(G2O_BASE_DIR ${EXTERN_DIR}/g2o)
   SET(G2O_HEADER_DIR ${G2O_BASE_DIR})
   SET(G2O_LIBRARY_DIR ${G2O_BASE_DIR}/lib)
ENDIF(G2O_NOT_FOUND)

# QGLViewer directories and project headers
SET(QGLVIEWER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/ext/qglviewer)
SET(QGLVIEWER_SOURCE_DIR ${EXTERN_DIR}/libQGLViewer)
# They will be placed automatically in a QGLViewer directory under here:
SET(QGLVIEWER_HEADER_DIR ${HEADER_DIR})

# Compounded G2O libraries
SET(G2O_LIBS
   ${G2O_CORE_LIBRARY}
   ${G2O_STUFF_LIBRARY}
   ${G2O_TYPES_SLAM3D}
   ${G2O_TYPES_SBA}
   ${G2O_SOLVER_CSPARSE}
   ${G2O_SOLVER_CSPARSE_EXTENSION}
)

LINK_DIRECTORIES(
   ${LIB_DIR}
   ${G2O_LIBRARY_DIR}
)
INCLUDE_DIRECTORIES(
   ${HEADER_DIR}
   ${LM_HEADER_DIR}
   ${G2O_HEADER_DIR}
   ${EIGEN3_INCLUDE_DIR}
   ${CSPARSE_INCLUDE_DIR}
)


################################################################################
# Print out dependencies so we can confirm they were found
################################################################################
message(STATUS "Searching for Dependencies...")
message(STATUS "-- Qt5              = ${Qt5OpenGL_LIBRARIES}")
message(STATUS "-- OpenGL           = ${OPENGL_LIBRARIES}")
message(STATUS "-- GLUT             = ${GLUT_LIBRARIES}")
message(STATUS "-- CSparse          = ${CSPARSE_LIBRARY}")
message(STATUS "-- G2O_CORE         = ${G2O_CORE_LIBRARY}")
message(STATUS "-- G2O_STUFF        = ${G2O_STUFF_LIBRARY}")
message(STATUS "-- G2O_SLAM3D       = ${G2O_TYPES_SLAM3D}")
message(STATUS "-- G2O_SBA          = ${G2O_TYPES_SBA}")
message(STATUS "-- G2O_CSPARSE      = ${G2O_SOLVER_CSPARSE}")
message(STATUS "-- G2O_CSPARSE_EXT  = ${G2O_SOLVER_CSPARSE_EXTENSION}")
message(STATUS "-- BLAS             = ${BLAS_LIBRARIES}")
message(STATUS "-- LAPACK           = ${LAPACK_LIBRARIES}")

# Build dependencies
ADD_SUBDIRECTORY(ext)

# Build MFG
ADD_SUBDIRECTORY(src)
