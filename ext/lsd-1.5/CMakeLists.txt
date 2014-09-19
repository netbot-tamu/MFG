PROJECT(lsd)

SET(CMAKE_C_FLAGS "-O3 -lm")

ADD_LIBRARY(${PROJECT_NAME}
   lsd.c
   lsd.h
)
