cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
cmake_policy(VERSION 3.10)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(NAME v4r)

IF (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
ENDIF()

project(${NAME} LANGUAGES CXX CUDA)

string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

add_compile_options(-Wall -Wextra -Wshadow)

# Hack to prefer linking against vulkan sdk
IF (DEFINED ENV{VULKAN_SDK})
    IF (DEFINED CMAKE_PREFIX_PATH)
        set(ORIG_PREFIX_PATH CMAKE_PREFIX_PATH)
    ENDIF()
    set(CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
ENDIF()

find_package(Vulkan REQUIRED)

IF (DEFINED ENV{VULKAN_SDK})
    IF (DEFINED ORIG_PREFIX_PATH)
        set(CMAKE_PREFIX_PATH ORIG_PREFIX_PATH)
    ELSE()
        unset(CMAKE_PREFIX_PATH)
    endif()
ENDIF()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

include_directories(external)

find_package(CUDAToolkit REQUIRED)

IF (NOT USE_BUNDLED)
    find_package(glm 0.9.9.8 CONFIG)
ENDIF()

IF (NOT glm_FOUND OR USE_BUNDLED)
    message(STATUS "Using bundled GLM")
    set(GLM_INCLUDE_OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/external/glm")
ELSE ()
    set(GLM_INCLUDE_INCLUDE_OVERRIDE "")
ENDIF()

IF (NOT USE_BUNDLED)
    find_package(ASSIMP 5.0.1 CONFIG)
    if (ASSIMP_FOUND)
        find_library(ASSIMP_LIBRARY_NAMES ${ASSIMP_LIBRARIES} PATH ${ASSIMP_LIBRARY_DIRS})
        set_target_properties(assimp::assimp PROPERTIES IMPORTED_LOCATION "${ASSIMP_LIBRARY_NAMES}")
    ENDIF()
ENDIF()

IF (NOT ASSIMP_FOUND OR USE_BUNDLED)
    message(STATUS "Using bundled assimp")
    set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "ASSIMP_BUILD_ASSIMP_TOOLS" FORCE)
    set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "ASSIMP_BUILD_TESTS" FORCE)
    add_subdirectory(external/assimp)
    include_directories(external/assimp/include)
ENDIF()

find_package(glfw3 3.3 QUIET)

add_subdirectory(src)
add_subdirectory(bin)
