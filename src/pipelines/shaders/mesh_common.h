#ifndef MESH_COMMON_H_INCLUDED
#define MESH_COMMON_H_INCLUDED

struct DrawInput {
    uint meshID;
};

struct MeshInfo {
    uint vertexOffset;
    uint indexOffset;
    uint indexCount;
};

#endif
