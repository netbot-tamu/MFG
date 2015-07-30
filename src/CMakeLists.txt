#PROJECT(mfg)

# TODO: make this an optional flag
SET(CMAKE_BUILD_TYPE debug)

# Subdirectories
INCLUDE_DIRECTORIES(
   utils
   features
   mfgcore
   gui
)

ADD_SUBDIRECTORY(utils)
# The 'features' module does not use Qt
SET(CMAKE_AUTOMOC OFF)
ADD_SUBDIRECTORY(features)
SET(CMAKE_AUTOMOC ON)
ADD_SUBDIRECTORY(mfgcore)
ADD_SUBDIRECTORY(nogui)
ADD_SUBDIRECTORY(gui)

