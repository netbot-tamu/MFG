# project is defined in the parent CMakeLists
project(mfglib)

set(SOURCES
   edge_line_vp_cam.cpp
   edge_line_vp_plane.cpp
   edge_point_plane.cpp
   edge_vnpt_cam.cpp
   glwidget.cpp
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
   #mfgthread.cpp
   #MfgTwoView.cpp
   mfg_utils.cpp
   #mfgwin.cpp
   #vertex_lineendpts.cpp
   #vertex_plane.cpp
   #vertex_vnpt.cpp
   #window.cpp
)

set(HEADERS
   glwidget.h
   mfgwin.h
)

set(HEADERS_NO_MOC
   edge_line_vp_cam.h
   edge_line_vp_plane.h
   edge_point_plane.h
   edge_vnpt_cam.h
   mfg.h
   mfg_utils.h
   vertex_lineendpts.h
   vertex_plane.h
   vertex_vnpt.h
   window.h
)


add_library(mfglib STATIC
   ${SOURCES}
   ${LSD_HEADERS}
   ${HEADERS_NO_MOC}
   ${MOC_SRC}
)

# Link the mfglib to all dependencies
target_link_libraries(mfglib 
   ${OpenGL_LIBRARIES}
   ${GLUT_LIBS}
   ${Qt5OpenGL_LIBRARIES}
   ${Qt5GUI_LIBRARIES}
   ${OpenCV_LIBS}
   ${Eigen3_LIBS}
)
qt5_use_modules(mfglib Core Widgets)


add_executable(mfg-qt main.cpp ${LSD_HEADERS} ${LM_HEADERS})

# Link the mfg-qt executable to all dependencies
#link_directories(../lib)
target_link_libraries(mfg-qt 
   mfglib
   ${OpenGL_LIBRARIES}
   ${GLUT_LIBS}
   ${Qt5OpenGL_LIBRARIES}
   #${Qt5GUI_LIBRARIES}
   ${OpenCV_LIBS}
   ${Eigen3_LIBS}
   #${CMAKE_SOURCE_DIR}/lib/libmfglib.so
)
qt5_use_modules(mfg-qt Core Gui Widgets OpenGL)


install(FILES ${HEADERS_NO_MOC} DESTINATION ${CMAKE_SOURCE_DIR}/includes/mfg)
install(TARGETS mfglib DESTINATION ${CMAKE_SOURCE_DIR}/lib)
install(TARGETS mfg-qt DESTINATION ${CMAKE_SOURCE_DIR}/bin)
