include_directories(.)
macro(make_library target source)
    add_library(${target} ${source})
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endmacro(make_library) 

FILE(GLOB SOURCES *.c)
set (SOURCE_FILES ${SOURCES})
set (CMAKE_C_FLAGS "-Wall -O2 -fPIC")

add_library(tap ${SOURCE_FILES})
