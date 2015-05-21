PROJECT(deps)

################################################################################
# Build simple dependencies first
################################################################################
ADD_SUBDIRECTORY(levmar-2.6)
ADD_SUBDIRECTORY(lsd-1.5)

# Special policy handling for G2O
IF(POLICY CMP0026)
CMAKE_POLICY(SET CMP0026 OLD)
ENDIF(POLICY CMP0026)

IF(POLICY CMP0042)
CMAKE_POLICY(SET CMP0042 OLD)
ENDIF(POLICY CMP0042)

# We want to skip the extra apps and examples to speed the build up for G2O.
IF(G2O_NOT_FOUND)
   MESSAGE(STATUS "G2O was not found, will build a local version")

   SET(G2O_BUILD_APPS OFF)
   SET(G2O_BUILD_EXAMPLES OFF)
   
   # this does not work - it gets overwritten by the lower level CMake
   SET(g2o_SOURCE_DIR ${mfg_SOURCE_DIR} CACHE STRING "G2O Source Directory")
   ADD_SUBDIRECTORY(g2o)
ENDIF(G2O_NOT_FOUND)


################################################################################
# Build QGLViewer Dependency...This one takes a little bit of setup to work 
# properly on OS X because clang++ does not link appropriately with the 
# particular combination of cmake and qmake being utilized.
################################################################################
INCLUDE(ExternalProject)
SET(QT_INCLUDE_PATHS
   ${Qt5Core_INCLUDE_DIRS}
   ${Qt5Widgets_INCLUDE_DIRS}
)

# Set some variables necessary for compilation on OS X (tested on 10.9).
IF(APPLE)
   # This function will convert the list of Qt5Widgets_INCLUDE_DIRS to a string
   # representation usable by qmake.
   function(JOIN VALUES GLUE OUTPUT)
      string (REGEX REPLACE "([^\\]|^);" "\\1${GLUE}" _TMP_STR "${VALUES}")
      string (REGEX REPLACE "[\\](.)" "\\1" _TMP_STR "${_TMP_STR}") #fixes escaping
      set (${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
   endfunction()

   JOIN("${Qt5Widgets_INCLUDE_DIRS} ${Qt5Xml_INCLUDE_DIRS}" " " QMAKE_QT_INCLUDE_DIRS)

   # Some necessary options for OS X (tested on 10.9 Mavericks):
   #  - use libc++
   #  - set the correct Qt include directories
   SET(QGLVIEWER_QMAKE_OPTIONS_OSX
      QMAKE_CXXFLAGS+=-stdlib=libc++
      QMAKE_LFLAGS+=-stdlib=libc++
      INCLUDEPATH+=${QMAKE_QT_INCLUDE_DIRS}
   )
ELSE(APPLE)
   SET(QGLVIEWER_QMAKE_OPTIONS_OSX "")
ENDIF(APPLE)

SET(QGLVIEWER_QMAKE_OPTIONS
   # We only need to build the main library, not the executables, so skip the
   # root level build from ${QGLVIEWER_SRC_DIR}/libQGLViewer.pro.
   ${QGLVIEWER_SOURCE_DIR}/QGLViewer/QGLViewer.pro
   # Generate a static library
   QGLVIEWER_STATIC=yes
   # PREFIX is the location of the build tree
   PREFIX=${QGLVIEWER_BINARY_DIR}
   # LIB_DIR is the location where the library will be created
   LIB_DIR=${LIB_DIR}
   # INCLUDE_DIR is the location where the includes will be created
   INCLUDE_DIR=${QGLVIEWER_HEADER_DIR}
   # Include platform-specific options
   ${QGLVIEWER_QMAKE_OPTIONS_OSX}
)

# This is the actual build command for QGLViewer
ExternalProject_Add(qglviewer
   PREFIX ${QGLVIEWER_BINARY_DIR}
   # do not download, as this is already downloaded as a git submodule
   DOWNLOAD_COMMAND ""
   CONFIGURE_COMMAND qmake ${QGLVIEWER_QMAKE_OPTIONS}
   BUILD_COMMAND make
)

