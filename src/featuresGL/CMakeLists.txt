project(featuresGL)

set(SOURCES
# currently, the edges are not being used in visualization, and have no symbols
   #edge_line_vp_cam_gl.cpp
   #edge_line_vp_plane_gl.cpp
   #edge_point_plane_gl.cpp
   #edge_vnpt_cam_gl.cpp
   vertex_lineendpts_gl.cpp
   vertex_plane_gl.cpp
   vertex_vnpt_gl.cpp
)

set(HEADERS
   edge_line_vp_cam_gl.h
   edge_line_vp_plane_gl.h
   edge_point_plane_gl.h
   edge_vnpt_cam_gl.h
   vertex_lineendpts_gl.h
   vertex_plane_gl.h
   vertex_vnpt_gl.h
)

add_library(${PROJECT_NAME} STATIC
   ${SOURCES}
)

target_link_libraries(${PROJECT_NAME}
   ${OPENGL_LIBRARIES}
   ${GLUT_LIBRARIES}
   utils
   features
)

# Install these files to the same directory as the base features
install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/includes/mfg/features)
#install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib)
