# project is defined in the parent CMakeLists
project(mfglib)

set(SOURCES
   mfg-ba-g2o.cpp
   mfg-ba-pt-g2o.cpp
   mfg-ba-ptln.cpp
   mfg-ba-ptlnvp.cpp
   mfg-ba-ptlnvp-relative.cpp
   mfg-ba-pt-relative.cpp
   mfg-ba-pts.cpp
   mfg.cpp
   MfgMap.cpp
   mfg-pos-ba.cpp
   MfgSingleView.cpp
   mfgthread.cpp
   MfgTwoView.cpp
   mfg_utils.cpp
   mfgwin.cpp
   window.cpp
)

set(HEADERS
   mfgwin.h
)

set(HEADERS_NO_MOC
   mfg.h
   mfg_utils.h
   window.h
)


add_library(mfglib STATIC
   ${SOURCES}
   ${LSD_HEADERS}
   ${HEADERS_NO_MOC}
   ${MOC_SRC}
)

# Link the mfglib to all dependencies
target_link_libraries(mfglib ${OpenCV_LIBS} ${Eigen3_LIBS})
qt5_use_modules(mfglib Core Widgets)


add_executable(mfg-qt main.cpp ${LSD_HEADERS} ${LM_HEADERS})

# Link the mfg-qt executable to all dependencies
#link_directories(../lib)
target_link_libraries(mfg-qt 
   ${OpenCV_LIBS}
   ${Eigen3_LIBS}
   ${CMAKE_SOURCE_DIR}/lib/libmfglib.so
   #msglib
)
qt5_use_modules(mfg-qt Core Widgets)


install(FILES ${HEADERS_NO_MOC} DESTINATION ${CMAKE_SOURCE_DIR}/includes/mfg)
install(TARGETS mfglib DESTINATION ${CMAKE_SOURCE_DIR}/lib)
install(TARGETS mfg-qt DESTINATION ${CMAKE_SOURCE_DIR}/bin)
