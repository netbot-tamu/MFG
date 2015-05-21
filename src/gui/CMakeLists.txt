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


add_executable(${PROJECT_NAME}
   main.cpp
   ${SOURCES}
   ${LSD_HEADERS}
   ${LM_HEADERS}
   ${MOC_SRC}
)

target_link_libraries(${PROJECT_NAME}
   ${OPENGL_LIBRARIES}
   ${GLUT_LIBRARIES}
   ${G2O_LIBS}
   ${OpenCV_LIBS}

   utils
   features
   featuresGL
   mfgcore
)

qt5_use_modules(${PROJECT_NAME}
   Core
   Widgets
   OpenGL
   Gui
)

#install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/includes/mfg/gui)
#install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/bin)
