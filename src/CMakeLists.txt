PROJECT(mfg-src)

# Subdirectories
INCLUDE_DIRECTORIES(
   utils
   features
   featuresGL
   mfgcore
   gui
)

ADD_SUBDIRECTORY(utils)
# The 'features' modules does not use Qt
SET(CMAKE_AUTOMOC OFF)
ADD_SUBDIRECTORY(features)
ADD_SUBDIRECTORY(featuresGL)
SET(CMAKE_AUTOMOC ON)
ADD_SUBDIRECTORY(mfgcore)
ADD_SUBDIRECTORY(nogui)
ADD_SUBDIRECTORY(gui)

