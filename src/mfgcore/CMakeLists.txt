# project is defined in the parent CMakeLists
project(mfgcore)

# TODO: make this an optional flag
set(CMAKE_BUILD_TYPE debug)

set(SOURCES
   est3dpt.cpp
   estfundm.cpp
   estfundm_g2o.cpp
   estfundm_helper.cpp
   export.cpp
   mfg-ba-g2o.cpp
   mfg-ba-pt-g2o.cpp
   mfg.cpp
   mfgthread.cpp
   mfgutils.cpp
   twoview.cpp
   view.cpp
)

set(HEADERS
   export.h
   mfg.h
   mfgutils.h
   twoview.h
   view.h
)

include_directories(
   ${LM_HEADER_DIR}
)

add_library(${PROJECT_NAME} STATIC
   ${SOURCES}
   ${LSD_HEADERS}
   ${LM_HEADERS}
   ${MOC_SRC}
)

# Link the ${PROJECT_NAME} to all dependencies
target_link_libraries(${PROJECT_NAME}
   # The following are NOT needed on OS X, but ARE needed on Linux Mint/Ubuntu
   ${OPENGL_LIBRARIES}
   ${G2O_CORE_LIBRARY}
   ${G2O_STUFF_LIBRARY}
   ${G2O_TYPES_SLAM3D}
   ${G2O_TYPES_SBA}
   ${G2O_SOLVER_CSPARSE}
   ${G2O_SOLVER_CSPARSE_EXTENSION}

   # These are needed regardless of OS
   ${Eigen3_LIBS}
   ${CSPARSE_LIBRARY}
   ${OpenCV_LIBS}

   levmar
   lsd
   utils
   features
)

qt5_use_modules(${PROJECT_NAME}
   Core
   Widgets
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/includes/mfg/core)
#install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib)
