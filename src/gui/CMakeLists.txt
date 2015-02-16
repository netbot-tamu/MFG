# project is defined in the parent CMakeLists
project(mfg-gui)

set(SOURCES
   glwidget.cpp
   window.cpp
)

set(HEADERS
   glwidget.h
   window.h
)


#add_library(${PROJECT_NAME} STATIC
add_executable(${PROJECT_NAME}
   main.cpp
   ${SOURCES}
   ${LSD_HEADERS}
   ${LM_HEADERS}
   ${MOC_SRC}
)

# Link the ${PROJECT_NAME} to all dependencies
target_link_libraries(${PROJECT_NAME}
   ${OPENGL_LIBRARIES}
   ${GLUT_LIBS}
   ${G2O_CORE_LIBRARY}
   ${G2O_STUFF_LIBRARY}
   ${G2O_TYPES_SLAM3D}
   ${G2O_TYPES_SBA}
   ${G2O_SOLVER_CSPARSE}
   ${G2O_SOLVER_CSPARSE_EXTENSION}
   ${Qt5OpenGL_LIBRARIES}
   ${Qt5GUI_LIBRARIES}
   ${OpenCV_LIBS}

   utils
   features
   mfgcore
)

qt5_use_modules(${PROJECT_NAME}
   Core
   Widgets
)

#install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/includes/mfg/gui)
#install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/bin)
