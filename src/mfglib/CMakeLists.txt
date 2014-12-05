# project is defined in the parent CMakeLists
project(mfglib)

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
   consts.h
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
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib)
