set(SHADER_FILES
    texture_unlit.vert
    texture_unlit.frag
    color_unlit.vert
    color_unlit.frag
)

find_program(GLSLC glslc)

set(COMPILED_SHADERS "")
FOREACH(SHADER_SRC ${SHADER_FILES})
    set(COMPILED_SHADER "${SHADER_SRC}.spv")
    list(APPEND COMPILED_SHADERS "${COMPILED_SHADER}")

    set(SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC}")
    add_custom_command(
        OUTPUT ${COMPILED_SHADER}
        COMMAND ${GLSLC} -c ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
    )
ENDFOREACH()

add_custom_target(compile_shaders DEPENDS ${COMPILED_SHADERS})
