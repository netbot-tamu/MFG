#PROJECT(mfg)

# TODO: make this an optional flag
SET(CMAKE_BUILD_TYPE debug)

# Subdirectories
INCLUDE_DIRECTORIES(
   utils
   features
   mfglib
   gui
)

ADD_SUBDIRECTORY(utils)
ADD_SUBDIRECTORY(features)
ADD_SUBDIRECTORY(mfglib)
ADD_SUBDIRECTORY(gui)

