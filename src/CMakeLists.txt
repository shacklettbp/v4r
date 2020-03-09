include_directories(${ASSIMP_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})

get_directory_property( DIR_DEFINES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS )
list(TRANSFORM DIR_DEFINES PREPEND -D)
list(JOIN DIR_DEFINES " " DEFINE_STR)

add_custom_command(
    OUTPUT dispatch/dispatch.template
    COMMAND mkdir -p dispatch && ${CMAKE_CXX_COMPILER} -x c -E -P -nostdinc ${DEFINE_STR} ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template -o dispatch/dispatch.template
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template
)

add_custom_command(
    OUTPUT dispatch/dispatch_device_impl.hpp
           dispatch/dispatch_device_impl.cpp
           dispatch/dispatch_instance_impl.hpp
           dispatch/dispatch_instance_impl.cpp
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/gen_vk_dispatch.py ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template dispatch
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template
            ${CMAKE_SOURCE_DIR}/scripts/gen_vk_dispatch.py
)

add_custom_target( generate_vk_dispatch DEPENDS
    dispatch/dispatch_device_impl.hpp
    dispatch/dispatch_device_impl.cpp
    dispatch/dispatch_instance_impl.hpp
    dispatch/dispatch_instance_impl.cpp
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/dispatch)

add_library(v4r SHARED
    v4r.cpp
    dispatch.hpp
    dispatch.cpp
)

add_dependencies(v4r generate_vk_dispatch)

target_link_libraries(v4r ${XCB_LIBRARIES} ${Vulkan_LIBRARY} ${ASSIMP_LIBRARIES} Threads::Threads)
