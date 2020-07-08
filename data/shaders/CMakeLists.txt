set(VERTEX_SHADER uber.vert)
set(FRAGMENT_SHADER uber.frag)
set(COMMON_SHADERS
    brdf.glsl)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             shaders_cfg.json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             ../../scripts/gen_shader_defines.py)

set(CFG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/shaders_cfg.json")
set(VERTEX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${VERTEX_SHADER}")
set(FRAGMENT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${FRAGMENT_SHADER}")

execute_process(
    COMMAND ../scripts/gen_shader_defines.py "${CFG_FILE}" "--cmake"
    OUTPUT_VARIABLE SHADER_DEFINES
    ERROR_VARIABLE CFG_ERROR
    RESULT_VARIABLE SHADER_SUCCESS)

if (NOT ${SHADER_SUCCESS} EQUAL 0)
    message(FATAL_ERROR "Failed to generate shader permutations ${CFG_ERROR}")
endif()

find_program(GLSLC glslc)

set(COMPILED_SHADERS "")

function(add_shader SHADER_NAME SHADER_SRC COMPILER_ARGS)
    set(COMPILED_SHADER "${SHADER_NAME}.spv")
    list(APPEND COMPILED_SHADERS "${COMPILED_SHADER}")
    set(COMPILED_SHADERS "${COMPILED_SHADERS}" PARENT_SCOPE)

    add_custom_command(
        OUTPUT
            ${COMPILED_SHADER}
        COMMAND
            ${GLSLC} -c ${SHADER_SRC} -o ${COMPILED_SHADER} ${COMPILER_ARGS}
        DEPENDS
            ${SHADER_SRC} ${COMMON_SHADERS}
    )
endfunction()

FOREACH(CFG_STR ${SHADER_DEFINES})
    separate_arguments(CFG_LIST UNIX_COMMAND "${CFG_STR}")
    list(GET CFG_LIST 0 SHADER_NAME)
    list(SUBLIST CFG_LIST 1 -1 SHADER_ARGS)

    add_shader("${SHADER_NAME}.vert" "${VERTEX_SRC}" "${SHADER_ARGS}")
    add_shader("${SHADER_NAME}.frag" "${FRAGMENT_SRC}" "${SHADER_ARGS}")
ENDFOREACH()

add_custom_target(compile_shaders DEPENDS ${COMPILED_SHADERS})
