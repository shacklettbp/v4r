#include <v4r/preprocess.hpp>
#include <meshoptimizer.h>

#include <fstream>

#include <glm/gtc/type_precision.hpp>

#include "asset_load.hpp"
#include "scene.hpp"
#include "utils.hpp"

using namespace std;

namespace v4r {

static constexpr int num_meshlet_vertices = 64;
static constexpr int num_meshlet_triangles = 126;
static constexpr int num_meshlets_per_chunk = 32;

struct SceneData {
    SceneDescription depthDesc;
    SceneDescription rgbDesc;
};

using DepthPipeline = Unlit<RenderOutputs::Depth, DataSource::None>;
using RGBPipeline = Unlit<RenderOutputs::Color, DataSource::Texture>;

static SceneData parseSceneData(string_view gltf_path)
{
    // FIXME make this all configurable

    SceneDescription depth_desc =
        LoaderImpl::create<DepthPipeline::Vertex,
                           DepthPipeline::MaterialParams>().parseScene(
                               gltf_path, glm::mat4(1.f));

    SceneDescription rgb_desc =
        LoaderImpl::create<RGBPipeline::Vertex,
                           RGBPipeline::MaterialParams>().parseScene(
                               gltf_path, glm::mat4(1.f));

    return SceneData {
        depth_desc,
        rgb_desc,
    };
}

ScenePreprocessor::ScenePreprocessor(string_view gltf_path)
    : scene_data_(new SceneData(parseSceneData(gltf_path)))
{}

struct Meshlet {
    glm::vec3 center;
    float radius;
    glm::i8vec3 coneAxis;
    int8_t coneCutoff;

    uint32_t bufferOffset;
    uint8_t vertexCount;
    uint8_t triangleCount;
    uint32_t pad[1];
};

struct MeshChunk {
    glm::vec3 center;
    float radius;
    
    uint32_t startIndex;
    uint32_t numMeshlets;
    uint32_t pad[2];
};

template <typename VertexType>
struct ProcessedMesh {
    vector<VertexType> vertices;
    vector<uint32_t> indices;

    vector<uint32_t> meshletBuffer;
    vector<Meshlet> meshlets;
    vector<MeshChunk> chunks;
};

static Meshlet makeMeshlet(const meshopt_Meshlet &src,
                           const meshopt_Bounds &bounds,
                           uint32_t offset)
{
    return Meshlet {
        glm::make_vec3(bounds.center),
        bounds.radius,
        glm::make_vec3<int8_t>(bounds.cone_axis_s8),
        bounds.cone_cutoff_s8,
        offset,
        src.vertex_count,
        src.triangle_count,
        {},
    };
}

static void appendMeshlet(vector<uint32_t> &meshlet_buffer,
                          const meshopt_Meshlet &meshlet)
{
    // Vertices
    for (uint8_t vert_idx = 0; vert_idx < meshlet.vertex_count; vert_idx++) {
        meshlet_buffer.push_back(meshlet.vertices[vert_idx]);
    }

    uint32_t index_offset = meshlet_buffer.size();
    meshlet_buffer.resize((meshlet.triangle_count * 3 + 3) / 4);

    uint8_t *idx_buf =
        reinterpret_cast<uint8_t *>(&meshlet_buffer[index_offset]);

    // Indices
    for (uint8_t tri_idx = 0; tri_idx < meshlet.triangle_count; tri_idx++) {
        for (uint32_t i = 0; i < 3; i++) {
            *idx_buf = meshlet.indices[tri_idx][i];
            idx_buf++;
        }
    }
}

template <typename VertexType>
static pair<vector<uint32_t>, vector<Meshlet>> buildMeshlets(
    const vector<VertexType> &vertices,
    const vector<uint32_t> &indices)
{
    vector<meshopt_Meshlet> raw_meshlets(
            meshopt_buildMeshletsBound(indices.size(),
                                       num_meshlet_vertices,
                                       num_meshlet_triangles));

    uint32_t num_meshlets =
        meshopt_buildMeshlets(raw_meshlets.data(), indices.data(),
                              indices.size(), vertices.size(),
                              num_meshlet_vertices, num_meshlet_triangles);

    raw_meshlets.resize(num_meshlets);

    vector<uint32_t> meshlet_buffer;
    vector<Meshlet> meshlets;
    meshlets.reserve(raw_meshlets.size());

    for (const meshopt_Meshlet &raw_meshlet : raw_meshlets) {
        meshopt_Bounds bounds = meshopt_computeMeshletBounds(
            &raw_meshlet, &vertices[0].position.x,
            vertices.size(), sizeof(VertexType));

        uint32_t buffer_offset = meshlet_buffer.size();

        appendMeshlet(meshlet_buffer, raw_meshlet);

        meshlets.emplace_back(makeMeshlet(raw_meshlet, bounds, buffer_offset));
    }

    // FIXME pad meshlets to warp size?

    return { move(meshlet_buffer), move(meshlets) };
}

vector<MeshChunk> assignChunks(const vector<Meshlet> &meshlets)
{
    uint32_t num_chunks = meshlets.size() / num_meshlets_per_chunk;
    if (meshlets.size() % num_chunks != 0) {
        num_chunks++;
    }

    vector<MeshChunk> chunks;
    chunks.reserve(num_chunks);

    uint32_t cur_meshlet_idx = 0;
    // Assign meshlets linearly to chunks. This matches how meshoptimizer
    // currently assigns meshlets, but will change apparently.

    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        uint32_t cur_num_meshlets = min<uint32_t>(
                num_meshlets_per_chunk, meshlets.size() - cur_meshlet_idx);

        glm::vec3 centroid(0.f);

        for (uint32_t local_meshlet_idx = 0;
             local_meshlet_idx < cur_num_meshlets;
             local_meshlet_idx++) {
            const Meshlet &cur_meshlet =
                meshlets[cur_meshlet_idx + local_meshlet_idx];

            centroid += 1.f / float(cur_num_meshlets) * cur_meshlet.center;
        }

        float radius = 0.f;
        for (uint32_t local_meshlet_idx = 0;
             local_meshlet_idx < cur_num_meshlets;
             local_meshlet_idx++) {
            const Meshlet &cur_meshlet =
                meshlets[cur_meshlet_idx + local_meshlet_idx];

            float to_center = glm::distance(cur_meshlet.center, centroid);

            radius = max(radius, to_center + cur_meshlet.radius);
        }

        assert(radius > 0.f);

        chunks.push_back(MeshChunk {
            centroid,
            radius,
            cur_meshlet_idx,
            cur_num_meshlets,
            {},
        });

        cur_meshlet_idx += cur_num_meshlets;
    }

    return chunks;
}

template <typename VertexType>
ProcessedMesh<VertexType> processMesh(const VertexMesh<VertexType> &orig_mesh)
{
    const vector<VertexType> &orig_vertices = orig_mesh.vertices;
    const vector<uint32_t> &orig_indices = orig_mesh.indices;
    uint32_t num_indices = orig_indices.size();

    vector<uint32_t> index_remap(num_indices);
    size_t new_vertex_count =
        meshopt_generateVertexRemap(index_remap.data(), orig_indices.data(),
                                    num_indices, orig_vertices.data(),
                                    orig_vertices.size(), sizeof(VertexType));

    vector<uint32_t> new_indices(num_indices);
    vector<VertexType> new_vertices(new_vertex_count);

    meshopt_remapIndexBuffer(new_indices.data(), orig_indices.data(),
                             num_indices, index_remap.data());

    meshopt_remapVertexBuffer(new_vertices.data(), orig_vertices.data(),
                              num_indices, sizeof(VertexType),
                              index_remap.data());

	meshopt_optimizeVertexCache(new_indices.data(), new_indices.data(),
                                num_indices, new_vertex_count);

	meshopt_optimizeVertexFetch(new_vertices.data(), new_indices.data(),
                                num_indices, new_vertices.data(),
                                new_vertex_count, sizeof(VertexType));

    auto [meshlet_buffer, meshlets] = 
        buildMeshlets(new_vertices, new_indices);

    auto chunks = assignChunks(meshlets);

    return ProcessedMesh<VertexType> {
        move(new_vertices),
        move(new_indices),
        move(meshlet_buffer),
        move(meshlets),
        move(chunks),
    };
}

void ScenePreprocessor::dump(string_view out_path)
{
    vector<ProcessedMesh<DepthPipeline::Vertex>> depth_meshes;

    for (const auto &depth_mesh : scene_data_->depthDesc.getMeshes()) {
        auto mesh_ptr = reinterpret_cast<VertexMesh<DepthPipeline::Vertex> *>(
            depth_mesh.get());

        depth_meshes.emplace_back(
            processMesh<DepthPipeline::Vertex>(*mesh_ptr));
    }

    vector<ProcessedMesh<RGBPipeline::Vertex>> rgb_meshes;

    for (const auto &rgb_mesh : scene_data_->rgbDesc.getMeshes()) {
        auto mesh_ptr = reinterpret_cast<VertexMesh<RGBPipeline::Vertex> *>(
            rgb_mesh.get());

        rgb_meshes.emplace_back(
            processMesh<RGBPipeline::Vertex>(*mesh_ptr));
    }

    ofstream out(filesystem::path(out_path), ios::binary);
    auto write = [&](auto val) {
        out.write(reinterpret_cast<char *>(&val), sizeof(decltype(val)));
    };
    write(uint32_t(0x55555555));
}

template struct HandleDeleter<SceneData>;

}
