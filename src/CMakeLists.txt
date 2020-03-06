include_directories(${ASSIMP_INCLUDE_DIR})
add_library(v4r SHARED
    v4r.cpp
)
target_link_libraries(v4r ${XCB_LIBRARIES} ${Vulkan_LIBRARY} ${ASSIMP_LIBRARIES} Threads::Threads)
