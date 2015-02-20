# project is defined in the parent CMakeLists
project(mfgcore)

set(SOURCES
   est3dpt.cpp
   estfundm.cpp
   estfundm_helper.cpp
   export.cpp
   mfg-ba-g2o.cpp
   mfg.cpp
   mfgthread.cpp
   mfgutils.cpp
   optimize.cpp
   view.cpp
)

set(HEADERS
   export.h
   mfg.h
   mfgutils.h
   optimize.h
   view.h
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
   ${G2O_LIBS}

   # These are needed regardless of OS
   ${Eigen3_LIBS}
   ${CSPARSE_LIBRARY}
   ${OpenCV_LIBS}

   # Libraries built by this project
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
