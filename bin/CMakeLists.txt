add_executable(bench
    bench.cpp
)
target_link_libraries(bench v4r_headless v4r_debug)

add_executable(save_frame
    save_frame.cpp
)
target_link_libraries(save_frame v4r_headless v4r_debug)

add_executable(raw_geometry
    raw_geometry.cpp
)
target_link_libraries(raw_geometry v4r_headless v4r_debug)

add_executable(singlebench
    singlebench.cpp
)
target_link_libraries(singlebench v4r_headless)

add_executable(double_buffer
    double_buffer.cpp
)
target_link_libraries(double_buffer v4r_headless)

add_executable(lighting
    lighting.cpp
)
target_link_libraries(lighting v4r_headless v4r_debug)

add_executable(preprocess
    preprocess.cpp
)
target_link_libraries(preprocess v4r_preprocess)

if (TARGET v4r_display)
    add_executable(display
        display.cpp
    )
    target_link_libraries(display v4r_display v4r_debug)

    add_executable(fly
        fly.cpp
    )
    target_link_libraries(fly v4r v4r_display v4r_debug)
endif()
