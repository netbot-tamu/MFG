# project is defined in the parent CMakeLists
project(mfg-nogui)

add_executable(${PROJECT_NAME}
   main.cpp
   ${SOURCES}
)

target_link_libraries(${PROJECT_NAME}
   ${OPENGL_LIBRARIES}
   ${G2O_CORE_LIBRARY}
   ${G2O_STUFF_LIBRARY}
   ${G2O_TYPES_SLAM3D}
   ${G2O_TYPES_SBA}
   ${G2O_SOLVER_CSPARSE}
   ${G2O_SOLVER_CSPARSE_EXTENSION}
   ${OpenCV_LIBS}

   features
   utils
   mfgcore
)

qt5_use_modules(${PROJECT_NAME}
   Core
)

#install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib)

