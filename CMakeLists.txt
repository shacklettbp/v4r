cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
cmake_policy(VERSION 3.10)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(NAME v4r)

IF (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
ENDIF()

set(HEADLESS "On" CACHE STRING "Headless rendering")

project(${NAME} LANGUAGES CXX)

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

find_package(ASSIMP REQUIRED CONFIG)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

if (!HEADLESS)
    find_package(XCB REQUIRED)
    add_definitions(-DVK_USE_PLATFORM_XCB_KHR)
ENDIF()

find_package(glm REQUIRED CONFIG)

include_directories(include external)

add_subdirectory(src)
add_subdirectory(bin)
