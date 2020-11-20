#ifndef MESH_COMMON_H_INCLUDED
#define MESH_COMMON_H_INCLUDED

struct DrawInput {
    uint instanceID;
    uint chunkID;
};

struct FrustumBounds {
    vec4 sides;
    vec2 nearFar;
};

struct CullPushConstant {
    FrustumBounds frustumBounds;
    uint batchIdx;
    uint baseDrawID;
    uint numDrawCommands;
};

struct Meshlet {
    vec3 center;
    float radius;
    i8vec3 coneAxis;
    int8_t coneCutoff;

    uint32_t bufferOffset;
    uint8_t vertexCount;
    uint8_t triangleCount;
    uint32_t pad[1];
};

struct MeshChunk {
    vec3 center;
    float radius;
    
    uint32_t meshletOffset;
    uint32_t numMeshlets;
    uint32_t vertexOffset;
    uint32_t indexOffset;
};

struct MeshInfo {
    uint32_t chunkOffset;
    uint32_t numChunks;
    uint32_t numVertices;
    uint32_t numIndices;
    uint64_t vertexOffset;
    uint64_t indexOffset;
};

#endif
