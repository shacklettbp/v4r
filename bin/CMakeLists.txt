add_executable(viewer
    viewer.cpp
)
target_link_libraries(viewer v4r)

add_executable(bench
    bench.cpp
)
target_link_libraries(bench v4r)

add_executable(save_frame
    save_frame.cpp
)
target_link_libraries(save_frame v4r v4r_debug)

add_executable(singlebench
    singlebench.cpp
)
target_link_libraries(singlebench v4r)
