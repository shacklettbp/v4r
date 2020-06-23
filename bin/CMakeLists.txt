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
