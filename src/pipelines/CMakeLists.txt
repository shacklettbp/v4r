set(VERTEX_SHADER shaders/uber.vert)
set(FRAGMENT_SHADER shaders/uber.frag)
set(RAYGEN_SHADER shaders/uber.rgen)
set(CLOSEST_HIT_SHADER shaders/uber.rchit)

set(SHADER_DEPENDENCIES
    shaders/shader_common.h
    shaders/mesh_common.h
    shaders/brdf.glsl)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             pipeline_cfg.json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             ../../scripts/gen_pipelines.py)

set(CFG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/pipeline_cfg.json")
set(VERTEX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${VERTEX_SHADER}")
set(FRAGMENT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${FRAGMENT_SHADER}")
set(RAYGEN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${RAYGEN_SHADER}")
set(CLOSEST_HIT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${CLOSEST_HIT_SHADER}")

set(IFACE_DIR "${CMAKE_CURRENT_BINARY_DIR}/interface")
set(IMPL_DIR "${CMAKE_CURRENT_BINARY_DIR}/implementation")

file(MAKE_DIRECTORY "${IFACE_DIR}")
file(MAKE_DIRECTORY "${IMPL_DIR}")

execute_process(
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/gen_pipelines.py"
            "${CFG_FILE}" "${IFACE_DIR}" "${IMPL_DIR}" --cmake
    OUTPUT_VARIABLE SHADER_DEFINES
    ERROR_VARIABLE CFG_ERROR
    RESULT_VARIABLE SHADER_SUCCESS)

if (NOT ${SHADER_SUCCESS} EQUAL 0)
    message(FATAL_ERROR "Failed to generate shader permutations ${CFG_ERROR}")
endif()

find_program(GLSLC glslc)

set(COMPILED_SHADERS "")
file(MAKE_DIRECTORY "${SHADER_DIR}")

function(add_shader SHADER_NAME SHADER_SRC COMPILER_ARGS)
    set(COMPILED_SHADER "${SHADER_DIR}/${SHADER_NAME}.spv")
    list(APPEND COMPILED_SHADERS "${COMPILED_SHADER}")
    set(COMPILED_SHADERS "${COMPILED_SHADERS}" PARENT_SCOPE)

    add_custom_command(
        OUTPUT
            ${COMPILED_SHADER}
        COMMAND
            ${GLSLC} -c ${SHADER_SRC} -o ${COMPILED_SHADER} --target-env=vulkan1.1 ${COMPILER_ARGS}
        DEPENDS
            ${SHADER_SRC} ${SHADER_DEPENDENCIES}
    )
endfunction()

FOREACH(CFG_STR ${SHADER_DEFINES})
    separate_arguments(CFG_LIST UNIX_COMMAND "${CFG_STR}")
    list(GET CFG_LIST 0 SHADER_NAME)
    list(SUBLIST CFG_LIST 1 -1 SHADER_ARGS)

    add_shader("${SHADER_NAME}.vert" "${VERTEX_SRC}" "${SHADER_ARGS}")
    add_shader("${SHADER_NAME}.frag" "${FRAGMENT_SRC}" "${SHADER_ARGS}")
ENDFOREACH()

add_shader("meshcull.comp" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/meshcull.comp" "")
add_shader("uber.rgen" "${RAYGEN_SRC}" "")
add_shader("uber.rchit" "${CLOSEST_HIT_SRC}" "")

add_custom_target(compile_shaders DEPENDS ${COMPILED_SHADERS})
