# project is defined in the parent CMakeLists
project(features)

# TODO: make this an optional flag
set(CMAKE_BUILD_TYPE debug)

set(SOURCES
   edge_line_vp_cam.cpp
   edge_line_vp_plane.cpp
   edge_point_plane.cpp
   edge_vnpt_cam.cpp
   epnp.cpp
   epnp_interface.cpp
   linematch.cpp
   vanishpoint.cpp
   vertex_lineendpts.cpp
   vertex_plane.cpp
   vertex_vnpt.cpp
   vpcov_helper.cpp
)

set(HEADERS
   edge_line_vp_cam.h
   edge_line_vp_plane.h
   edge_point_plane.h
   edge_vnpt_cam.h
   epnp.h
   features2d.h
   features3d.h
   vertex_lineendpts.h
   vertex_plane.h
   vertex_vnpt.h
)

include_directories(
   ${LM_HEADER_DIR}
)

add_library(${PROJECT_NAME} STATIC
   ${SOURCES}
   ${LM_HEADERS}
   ${MOC_SRC}
)

target_link_libraries(${PROJECT_NAME}
   utils
)

qt5_use_modules(${PROJECT_NAME}
   Core
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/includes/mfg/features)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib)
