add_executable(viewer
    viewer.cpp
)
target_link_libraries(viewer v4r)

add_executable(bench
    bench.cpp
)
target_link_libraries(bench v4r)
