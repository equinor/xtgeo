include_directories(.)

macro(make_library target source)
    add_library(${target} ${source})
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endmacro(make_library)

FILE(GLOB SOURCES *.c)
set (SOURCE_FILES ${SOURCES})
if (MSVC)
   set (CMAKE_C_FLAGS "/Ox")
else()
	set (CMAKE_C_FLAGS "-Wall -O2 -fPIC -Wno-unused-but-set-variable")
endif()

add_library(cxtgeo STATIC ${SOURCE_FILES})
