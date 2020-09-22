#include <v4r/preprocess.hpp>
#include <meshoptimizer.h>

#include <fstream>

#include <glm/gtc/type_precision.hpp>
#include <glm/gtx/string_cast.hpp>

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

template <typename VertexType>
struct ProcessedMesh {
    vector<VertexType> vertices;
    vector<uint32_t> indices;

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
    meshlet_buffer.resize(meshlet_buffer.size() +
                          (meshlet.triangle_count * 3 + 3) / 4);

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
static vector<Meshlet> buildMeshlets(
    const vector<VertexType> &vertices,
    const vector<uint32_t> &indices,
    vector<uint32_t> &meshlet_buffer)
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

    vector<Meshlet> meshlets;
    meshlets.reserve(raw_meshlets.size());

    for (const meshopt_Meshlet &raw_meshlet : raw_meshlets) {
        meshopt_Bounds bounds = meshopt_computeMeshletBounds(
            &raw_meshlet, &vertices[0].position.x,
            vertices.size(), sizeof(VertexType));

        assert(bounds.radius != 0);

        uint32_t buffer_offset = meshlet_buffer.size();

        appendMeshlet(meshlet_buffer, raw_meshlet);

        meshlets.emplace_back(makeMeshlet(raw_meshlet, bounds, buffer_offset));
    }

    // FIXME pad meshlets to warp size?

    return meshlets;
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

        assert(cur_num_meshlets != 0);

        glm::vec3 centroid(0.f);
        uint32_t num_vertices = 0;
        uint32_t num_indices = 0;
        for (uint32_t local_meshlet_idx = 0;
             local_meshlet_idx < cur_num_meshlets;
             local_meshlet_idx++) {
            const Meshlet &cur_meshlet =
                meshlets[cur_meshlet_idx + local_meshlet_idx];

            centroid += cur_meshlet.center / float(cur_num_meshlets) ;

            num_vertices += cur_meshlet.vertexCount;
            num_indices += cur_meshlet.triangleCount * 3;
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
            num_vertices, // Updated to be global during serialization
            num_indices, // Updated to be global during serialization
        });

        cur_meshlet_idx += cur_num_meshlets;
    }

    return chunks;
}

template <typename VertexType>
ProcessedMesh<VertexType> processMesh(const VertexMesh<VertexType> &orig_mesh,
                                      vector<uint32_t> &meshlet_buffer)
{
    const vector<VertexType> &orig_vertices = orig_mesh.vertices;
    const vector<uint32_t> &orig_indices = orig_mesh.indices;
    uint32_t num_indices = orig_indices.size();

    vector<uint32_t> index_remap(orig_vertices.size());
    size_t new_vertex_count =
        meshopt_generateVertexRemap(index_remap.data(), orig_indices.data(),
                                    num_indices, orig_vertices.data(),
                                    orig_vertices.size(), sizeof(VertexType));

    vector<uint32_t> new_indices(num_indices);
    vector<VertexType> new_vertices(new_vertex_count);

    meshopt_remapIndexBuffer(new_indices.data(), orig_indices.data(),
                             num_indices, index_remap.data());

    meshopt_remapVertexBuffer(new_vertices.data(), orig_vertices.data(),
                              orig_vertices.size(), sizeof(VertexType),
                              index_remap.data());

    meshopt_optimizeVertexCache(new_indices.data(), new_indices.data(),
                                num_indices, new_vertex_count);

    new_vertex_count = meshopt_optimizeVertexFetch(new_vertices.data(),
                                                   new_indices.data(),
                                                   num_indices,
                                                   new_vertices.data(),
                                                   new_vertex_count,
                                                   sizeof(VertexType));
    new_vertices.resize(new_vertex_count);

    auto meshlets = buildMeshlets(new_vertices, new_indices, meshlet_buffer);
 
    auto chunks = assignChunks(meshlets);

    return ProcessedMesh<VertexType> {
        move(new_vertices),
        move(new_indices),
        move(meshlets),
        move(chunks),
    };
}

void ScenePreprocessor::dump(string_view out_path_name)
{
    const SceneDescription &depth_desc = scene_data_->depthDesc;
    const SceneDescription &rgb_desc = scene_data_->rgbDesc;

    vector<ProcessedMesh<DepthPipeline::Vertex>> depth_meshes;
    vector<uint32_t> depth_meshlet_buffer;
    for (const auto &depth_mesh : depth_desc.getMeshes()) {
        auto mesh_ptr = reinterpret_cast<VertexMesh<DepthPipeline::Vertex> *>(
            depth_mesh.get());

        depth_meshes.emplace_back(
            processMesh<DepthPipeline::Vertex>(*mesh_ptr,
                depth_meshlet_buffer));
    }

    vector<ProcessedMesh<RGBPipeline::Vertex>> rgb_meshes;
    vector<uint32_t> rgb_meshlet_buffer;
    for (const auto &rgb_mesh : rgb_desc.getMeshes()) {
        auto mesh_ptr = reinterpret_cast<VertexMesh<RGBPipeline::Vertex> *>(
            rgb_mesh.get());

        rgb_meshes.emplace_back(
            processMesh<RGBPipeline::Vertex>(*mesh_ptr,
                rgb_meshlet_buffer));
    }

    filesystem::path out_path(out_path_name);
    string basename = out_path.filename();

    ofstream out(out_path, ios::binary);
    auto write = [&](auto val) {
        out.write(reinterpret_cast<char *>(&val), sizeof(decltype(val)));
    };

    // FIXME material system needs to be redone. Don't actually know
    // the texture names at this point.
    auto write_materials = [&](const SceneDescription &desc) {
        uint32_t num_materials = desc.getMaterials().size();
        write(num_materials);
        for (uint32_t mat_idx = 0; mat_idx < num_materials; mat_idx++) {
            out << (basename + "_" + to_string(mat_idx) + ".ktx2");
        }
    };

    auto write_meshes = [&](auto meshes, auto meshlet_buffer) {
        uint32_t num_vertices = 0;
        uint32_t num_indices = 0;
        uint32_t num_meshlets = 0;
        uint32_t num_chunks = 0;

        vector<MeshInfo> mesh_infos;

        for (auto &mesh : meshes) {
            // FIXME doing this offset here seems hacky
            // Need to change all chunk offsets to be global
            // to the whole scene
            for (auto &chunk : mesh.chunks) {
                chunk.vertexOffset += num_vertices;
                chunk.indexOffset += num_indices;
                chunk.meshletOffset += num_meshlets;
            }

            uint32_t chunk_offset = num_chunks;
            assert(chunk_offset < 65536);
            assert(mesh.chunks.size() < 65536);
            mesh_infos.push_back(MeshInfo {
                uint16_t(chunk_offset),
                uint16_t(mesh.chunks.size()),
                num_vertices,
                num_indices,
                uint32_t(mesh.indices.size()),
            });

            num_vertices += mesh.vertices.size();
            num_indices += mesh.indices.size();
            num_meshlets += mesh.meshlets.size();
            num_chunks += mesh.chunks.size();
        }

        write(num_vertices);
        // Write all verticess
        for (auto &mesh : meshes) {
            out.write(reinterpret_cast<const char *>(mesh.vertices.data()),
                      mesh.vertices.size() * sizeof(
                          typename decltype(mesh.vertices)::value_type));
        }

        write(num_indices);
        // Write all indices
        for (auto &mesh : meshes) {
            out.write(reinterpret_cast<const char *>(mesh.indices.data()),
                      mesh.indices.size() * sizeof(uint32_t));
        }

        // Write meshlet buffer
        write(uint32_t(meshlet_buffer.size()));
        out.write(reinterpret_cast<const char *>(meshlet_buffer.data()),
                  meshlet_buffer.size() * sizeof(uint32_t));

        // Write meshlets
        write(num_meshlets);
        for (auto &mesh : meshes) {
            out.write(reinterpret_cast<const char *>(mesh.meshlets.data()),
                      mesh.meshlets.size() * sizeof(Meshlet));
        }

        // Write chunks
        write(num_chunks);
        for (auto &mesh : meshes) {
            out.write(reinterpret_cast<const char *>(mesh.chunks.data()),
                      mesh.chunks.size() * sizeof(MeshChunk));
        }

        // Write mesh infos
        write(uint32_t(mesh_infos.size()));
        out.write(reinterpret_cast<const char *>(mesh_infos.data()),
                  mesh_infos.size() * sizeof(MeshInfo));
    };

    auto write_instances = [&](const SceneDescription &desc) {
        const vector<InstanceProperties> &instances =
            desc.getDefaultInstances();
        write(uint32_t(instances.size()));
        out.write(reinterpret_cast<const char *>(instances.data()),
                  instances.size() * sizeof(InstanceProperties));
    };

    // Pad to 16 bytes
    auto write_pad = [&]() {
        static char pad_buffer[15] = { 0 };
        size_t cur_bytes = out.tellp();
        size_t align = cur_bytes % 16;
        if (align != 0) {
            out.write(pad_buffer, 16 - align);
        }
    };

    // Header: magic + depth offset + rgb offset (bytes from header)
    write(uint32_t(0x55555555));
    write(uint32_t(0));
    write(uint32_t(0)); // Rewrite later
    write_materials(depth_desc);
    write_pad();
    write_meshes(move(depth_meshes), move(depth_meshlet_buffer));
    write_pad();
    write_instances(depth_desc);
    write_pad();
    uint32_t rgb_offset = out.tellp() / sizeof(uint32_t);
    out.seekp(8);
    write(rgb_offset);
    out.seekp(rgb_offset * sizeof(uint32_t));
    write_materials(rgb_desc);
    write_pad();
    write_meshes(move(rgb_meshes), move(rgb_meshlet_buffer));
    write_pad();
    write_instances(rgb_desc);
    write_pad();

    out.close();
}

template struct HandleDeleter<SceneData>;

}
