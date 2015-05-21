# project is defined in the parent CMakeLists
project(mfg-nogui)

add_executable(${PROJECT_NAME}
   main.cpp
)

target_link_libraries(${PROJECT_NAME}
   ${G2O_LIBS}
   ${OpenCV_LIBS}

   utils
   features
   mfgcore
)

qt5_use_modules(${PROJECT_NAME}
   Core
)

#install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib)

