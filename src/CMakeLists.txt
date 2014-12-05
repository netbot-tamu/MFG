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
# The 'features' module does not use Qt
SET(CMAKE_AUTOMOC OFF)
ADD_SUBDIRECTORY(features)
SET(CMAKE_AUTOMOC ON)
ADD_SUBDIRECTORY(mfglib)
#ADD_SUBDIRECTORY(core)
ADD_SUBDIRECTORY(gui)

