include_directories(${ASSIMP_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})

add_definitions(-DVK_NO_PROTOTYPES -DGLM_FORCE_DEPTH_ZERO_TO_ONE)

# Build vulkan dispatch table
get_directory_property(DIR_DEFINES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS )
set(DEFINE_STR "")
FOREACH(f ${DIR_DEFINES})
    list(APPEND DEFINE_STR "-D${f}")
ENDFOREACH()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dispatch")

add_custom_command(
    OUTPUT dispatch/dispatch.template
    COMMAND ${CMAKE_CXX_COMPILER} -x c -E -P -nostdinc ${DEFINE_STR} "${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template" -o dispatch/dispatch.template
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template
)

add_custom_command(
    OUTPUT dispatch/dispatch_device_impl.hpp dispatch/dispatch_device_impl.cpp
           dispatch/dispatch_instance_impl.hpp dispatch/dispatch_instance_impl.cpp
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/gen_vk_dispatch.py
                ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template dispatch
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template
            ${CMAKE_SOURCE_DIR}/scripts/gen_vk_dispatch.py
)

add_custom_target( generate_vk_dispatch DEPENDS
    dispatch/dispatch_device_impl.hpp dispatch/dispatch_device_impl.cpp
    dispatch/dispatch_instance_impl.hpp dispatch/dispatch_instance_impl.cpp
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/dispatch)

# Build shaders
set(SHADER_DIR "${CMAKE_BINARY_DIR}/data/shaders/")
add_subdirectory("../data/shaders/" "${SHADER_DIR}")
add_definitions("-DSHADER_DIR=${SHADER_DIR}")

# Main library

add_library(v4r SHARED
    dispatch.hpp dispatch.cpp
    scene.hpp scene.cpp
    utils.hpp utils.cpp
    vk_utils.hpp vk_utils.cpp
    vulkan_config.hpp ../include/v4r/config.hpp
    vulkan_state.hpp vulkan_state.cpp
    ../include/v4r.hpp v4r.cpp
)

add_dependencies(v4r generate_vk_dispatch compile_shaders)

target_link_libraries(v4r Vulkan::Vulkan Threads::Threads ${ASSIMP_LIBRARIES})
