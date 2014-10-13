# project is defined in the parent CMakeLists
project(mfglib)

set(SOURCES
   edge_line_vp_cam.cpp
   edge_line_vp_plane.cpp
   edge_point_plane.cpp
   edge_vnpt_cam.cpp
   epnp.cpp
   epnp_interface.cpp
   est3dpt.cpp
   estfundm.cpp
   estfundm_g2o.cpp
   estfundm_helper.cpp
   export.cpp
   glwidget.cpp
   linematch.cpp
   mfg-ba-g2o.cpp
   mfg-ba-pt-g2o.cpp
   #mfg-ba-ptln.cpp
   #mfg-ba-ptlnvp.cpp
   #mfg-ba-ptlnvp-relative.cpp
   #mfg-ba-pt-relative.cpp
   #mfg-ba-pts.cpp
   mfg.cpp
   #MfgMap.cpp
   #mfg-pos-ba.cpp
   # TODO: guessing for below comments
   #MfgSingleView.cpp
   mfgthread.cpp
   #MfgTwoView.cpp
   mfg_utils.cpp
   #mfgwin.cpp
   TwoView.cpp
   vanishpoint.cpp
   vertex_lineendpts.cpp
   vertex_plane.cpp
   vertex_vnpt.cpp
   vpcov_helper.cpp
   view.cpp
   window.cpp
   settings.cpp
)

set(HEADERS
   consts.h
   edge_line_vp_cam.h
   edge_line_vp_plane.h
   edge_point_plane.h
   edge_vnpt_cam.h
   epnp.h
   glwidget.h
   mfg.h
   mfg_utils.h
   mfgwin.h
   vertex_lineendpts.h
   vertex_plane.h
   vertex_vnpt.h
   view.h
   window.h

   settings.h
)


#add_library(mfglib STATIC
add_executable(${PROJECT_NAME}
   main.cpp
   ${SOURCES}
   ${LSD_HEADERS}
   ${LM_HEADERS}
   ${MOC_SRC}
)

# Link the mfglib to all dependencies
target_link_libraries(${PROJECT_NAME}
   ${OPENGL_LIBRARIES}
   ${GLUT_LIBS}
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

   ${Qt5OpenGL_LIBRARIES}
   ${Qt5GUI_LIBRARIES}
   ${OpenCV_LIBS}
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


install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/includes/mfg)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib)
#install(TARGETS mfg-qt DESTINATION ${CMAKE_SOURCE_DIR}/bin)
