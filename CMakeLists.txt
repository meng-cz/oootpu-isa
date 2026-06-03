cmake_minimum_required(VERSION 3.20)

project(oootpu_isa LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Supported configurations" FORCE)
else()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)
endif()

file(GLOB_RECURSE ISA_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/isa/*.cpp"
)

file(GLOB_RECURSE FUNCSIM_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/funcsim/*.cpp"
)

add_library(oootpu_local STATIC
  ${ISA_SOURCES}
  ${FUNCSIM_SOURCES}
)

target_include_directories(oootpu_local
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/isa"
    "${CMAKE_CURRENT_SOURCE_DIR}/funcsim"
)

target_compile_features(oootpu_local PUBLIC cxx_std_23)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(oootpu_local
    PRIVATE
      $<$<CONFIG:Debug>:
        -O0
        -g3
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Wformat=2
        -Wundef
        -Wcast-qual
        -Wcast-align
        -Wnull-dereference
        -Wdouble-promotion
        -Wimplicit-fallthrough
        -Woverloaded-virtual
        -Wnon-virtual-dtor
        -Wold-style-cast
        -fsanitize=address,undefined,leak
      >
      $<$<CONFIG:Release>:
        -O3
      >
  )

  target_link_options(oootpu_local
    PRIVATE
      $<$<CONFIG:Debug>:
        -fsanitize=address,undefined,leak
      >
  )
elseif(MSVC)
  target_compile_options(oootpu_local
    PRIVATE
      $<$<CONFIG:Debug>:
        /Od
        /Zi
        /sdl
        /W4
        /WX
        /permissive-
      >
      $<$<CONFIG:Release>:
        /O2
      >
  )
endif()

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/example/*.cpp"
)

foreach(example_source IN LISTS EXAMPLE_SOURCES)
  get_filename_component(example_name "${example_source}" NAME_WE)

  add_executable("${example_name}" "${example_source}")
  target_link_libraries("${example_name}" PRIVATE oootpu_local)
  target_include_directories("${example_name}"
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}/isa"
      "${CMAKE_CURRENT_SOURCE_DIR}/funcsim"
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options("${example_name}"
      PRIVATE
        $<$<CONFIG:Debug>:
          -O0
          -g3
          -fno-omit-frame-pointer
          -fno-optimize-sibling-calls
          -Wall
          -Wextra
          -Wpedantic
          -Werror
          -Wconversion
          -Wsign-conversion
          -Wshadow
          -Wformat=2
          -Wundef
          -Wcast-qual
          -Wcast-align
          -Wnull-dereference
          -Wdouble-promotion
          -Wimplicit-fallthrough
          -Woverloaded-virtual
          -Wnon-virtual-dtor
          -Wold-style-cast
          -fsanitize=address,undefined,leak
        >
        $<$<CONFIG:Release>:
          -O3
        >
    )

    target_link_options("${example_name}"
      PRIVATE
        $<$<CONFIG:Debug>:
          -fsanitize=address,undefined,leak
        >
    )
  elseif(MSVC)
    target_compile_options("${example_name}"
      PRIVATE
        $<$<CONFIG:Debug>:
          /Od
          /Zi
          /sdl
          /W4
          /WX
          /permissive-
        >
        $<$<CONFIG:Release>:
          /O2
        >
    )
  endif()
endforeach()
