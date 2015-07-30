# project is defined in the parent CMakeLists
project(utils)

# TODO: make this an optional flag
set(CMAKE_BUILD_TYPE debug)

set(SOURCES
   random.cpp
   utils.cpp
   settings.cpp
)

set(HEADERS
   consts.h
   random.h
   utils.h
   settings.h
)


add_library(${PROJECT_NAME} STATIC
   ${SOURCES}
   ${MOC_SRC}
)

target_link_libraries(${PROJECT_NAME}
   ${OpenCV_LIBS}
)

qt5_use_modules(${PROJECT_NAME}
   Core
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/includes/mfg/utils)
#install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/lib)
