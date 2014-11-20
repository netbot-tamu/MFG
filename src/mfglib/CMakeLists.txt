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
#   ${OPENGL_LIBRARIES}
#   ${GLUT_LIBS}
   ${Eigen3_LIBS}
   ${CSPARSE_LIBRARY}
   ${G2O_CORE_LIBRARY}
   ${G2O_STUFF_LIBRARY}
   ${G2O_TYPES_SLAM3D}
   ${G2O_TYPES_SBA}
   ${G2O_SOLVER_CSPARSE}
   ${G2O_SOLVER_CSPARSE_EXTENSION}
   ${BLAS_LIBRARIES}
   ${LAPACK_LIBRARIES}

   # Levmar 2.6 is included in this project for simplicity
   levmar

   # LSD (Line Segment Detector) 1.5 is included in this project for simplicity
   lsd

#   ${Qt5OpenGL_LIBRARIES}
#   ${Qt5GUI_LIBRARIES}
   ${OpenCV_LIBS}

   features
   utils
)

qt5_use_modules(${PROJECT_NAME}
   Core
   Widgets
)


#add_executable(mfg-qt main.cpp
#   ${LSD_HEADERS}
#   ${LM_HEADERS}
#)

# Link the mfg-qt executable to all dependencies
#link_directories(../lib)
#target_link_libraries(mfg-qt
#   #${OPENGL_LIBRARIES}
#   #${GLUT_LIBS}
#   #${Qt5OpenGL_LIBRARIES}
#   #${Qt5GUI_LIBRARIES}
#   #${OpenCV_LIBS}
#   #${Eigen3_LIBS}
#   #${CMAKE_SOURCE_DIR}/lib/libmfglib.so
#   mfglib
#)
#qt5_use_modules(mfg-qt Core Gui Widgets OpenGL)


install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/includes/mfg/core)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib)
#install(TARGETS mfg-qt DESTINATION ${CMAKE_SOURCE_DIR}/bin)
