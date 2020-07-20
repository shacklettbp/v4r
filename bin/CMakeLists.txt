add_executable(bench
    bench.cpp
)
target_link_libraries(bench ${V4R_HEADLESS_LIBS} v4r_debug)

add_executable(save_frame
    save_frame.cpp
)
target_link_libraries(save_frame ${V4R_HEADLESS_LIBS} v4r_debug)

add_executable(raw_geometry
    raw_geometry.cpp
)
target_link_libraries(raw_geometry ${V4R_HEADLESS_LIBS} v4r_debug)

add_executable(singlebench
    singlebench.cpp
)
target_link_libraries(singlebench ${V4R_HEADLESS_LIBS})

add_executable(double_buffer
    double_buffer.cpp
)
target_link_libraries(double_buffer ${V4R_HEADLESS_LIBS})

add_executable(lighting
    lighting.cpp
)
target_link_libraries(lighting ${V4R_HEADLESS_LIBS} v4r_debug)

if (TARGET v4r_display)
    add_executable(display
        display.cpp
    )
    target_link_libraries(display v4r v4r_display v4r_debug)

    add_executable(fly
        fly.cpp
    )
    target_link_libraries(fly v4r v4r_display v4r_debug)
endif()
