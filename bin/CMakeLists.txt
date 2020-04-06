add_executable(viewer
    viewer.cpp
)

target_link_libraries(viewer ${V4R_HEADLESS_LIBS})

add_executable(bench
    bench.cpp
)
target_link_libraries(bench ${V4R_HEADLESS_LIBS})

add_executable(save_frame
    save_frame.cpp
)
target_link_libraries(save_frame ${V4R_HEADLESS_LIBS} v4r_debug)

add_executable(singlebench
    singlebench.cpp
)
target_link_libraries(singlebench ${V4R_HEADLESS_LIBS})
