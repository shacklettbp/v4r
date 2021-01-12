#include <v4r/preprocess.hpp>
#include <meshoptimizer.h>

#include <fstream>

#include <glm/gtc/type_precision.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/norm.hpp>

#include "asset_load.hpp"
#include "scene.hpp"
#include "utils.hpp"

using namespace std;

namespace v4r {

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
                                       VulkanConfig::num_meshlet_vertices,
                                       VulkanConfig::num_meshlet_triangles));

    uint32_t num_meshlets =
        meshopt_buildMeshlets(raw_meshlets.data(), indices.data(),
                              indices.size(), vertices.size(),
                              VulkanConfig::num_meshlet_vertices,
                              VulkanConfig::num_meshlet_triangles);

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
    uint32_t num_chunks =
        (meshlets.size() + VulkanConfig::num_meshlets_per_chunk - 1) /
        VulkanConfig::num_meshlets_per_chunk;

    vector<MeshChunk> chunks;
    chunks.reserve(num_chunks);

    uint32_t cur_meshlet_idx = 0;
    uint32_t num_vertices = 0;
    uint32_t num_indices = 0;

    // Assign meshlets linearly to chunks. This matches how meshoptimizer
    // currently assigns meshlets, but will change apparently.
    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        uint32_t cur_num_meshlets = min<uint32_t>(
                VulkanConfig::num_meshlets_per_chunk,
                meshlets.size() - cur_meshlet_idx);

        assert(cur_num_meshlets != 0);

        uint32_t chunk_index_offset = num_indices;

        glm::vec3 centroid(0.f);
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
            chunk_index_offset, // Updated to be global offset
            0, // Padding
        });

        // FIXME hack
        chunks.back().numMeshlets = num_indices - chunk_index_offset;

        cur_meshlet_idx += cur_num_meshlets;
    }

    assert(cur_meshlet_idx == meshlets.size());

    return chunks;
}

template <typename VertexType>
static vector<uint32_t> filterDegenerateTriangles(
    const vector<VertexType> &vertices,
    const vector<uint32_t> &orig_indices)
{
    vector<uint32_t> new_indices;
    new_indices.reserve(orig_indices.size());

    uint32_t num_indices = orig_indices.size();
    uint32_t tri_align = orig_indices.size() % 3;
    if (tri_align != 0) {
        cerr << "Warning: non multiple of 3 indices in mesh" << endl;
        num_indices -= tri_align;
    }
    assert(orig_indices.size() % 3 == 0);

    for (uint32_t i = 0; i < num_indices;) {
        uint32_t a_idx = orig_indices[i++];
        uint32_t b_idx = orig_indices[i++];
        uint32_t c_idx = orig_indices[i++];

        glm::vec3 a = vertices[a_idx].position;
        glm::vec3 b = vertices[b_idx].position;
        glm::vec3 c = vertices[c_idx].position;

        glm::vec3 ab = a - b;
        glm::vec3 bc = b - c;
        float check = glm::length2(glm::cross(ab, bc));

        if (check < 1e-20f) {
            continue;
        }

        new_indices.push_back(a_idx);
        new_indices.push_back(b_idx);
        new_indices.push_back(c_idx);
    }

    uint32_t num_degenerate = orig_indices.size() - new_indices.size();

    if (num_degenerate > 0) {
        cout << "Filtered: " << num_degenerate
             << " degenerate triangles" << endl;
    }

    return new_indices;
}

template <typename VertexType>
optional<ProcessedMesh<VertexType>> processMesh(
    const VertexMesh<VertexType> &orig_mesh,
    vector<uint32_t> &meshlet_buffer)
{
    const vector<VertexType> &orig_vertices = orig_mesh.vertices;
    const vector<uint32_t> &orig_indices = orig_mesh.indices;

    vector<uint32_t> filtered_indices =
        filterDegenerateTriangles(orig_vertices, orig_indices);

    if (filtered_indices.size() == 0) {
        cerr << "Warning: removing entire degenerate mesh" << endl;
        return optional<ProcessedMesh<VertexType>>();
    }

    uint32_t num_indices = filtered_indices.size();

    vector<uint32_t> index_remap(orig_vertices.size());
    size_t new_vertex_count =
        meshopt_generateVertexRemap(index_remap.data(),
                                    filtered_indices.data(),
                                    num_indices, orig_vertices.data(),
                                    orig_vertices.size(), sizeof(VertexType));

    vector<uint32_t> new_indices(num_indices);
    vector<VertexType> new_vertices(new_vertex_count);

    meshopt_remapIndexBuffer(new_indices.data(), filtered_indices.data(),
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

template <typename VertexType>
struct ProcessedGeometry {
    vector<ProcessedMesh<VertexType>> meshes;
    vector<uint32_t> meshletBuffer;
    vector<uint32_t> meshIDRemap;
    vector<MeshInfo> meshInfos;
    uint32_t totalVertices;
    uint32_t totalIndices;
    uint32_t totalMeshlets;
    uint32_t totalChunks;
};

template <typename VertexType>
static ProcessedGeometry<VertexType> processGeometry(
    const SceneDescription &desc)
{
    const auto &orig_meshes = desc.getMeshes();
    vector<ProcessedMesh<VertexType>> processed_meshes;
    vector<uint32_t> meshlet_buffer;

    vector<uint32_t> mesh_id_remap(desc.getMeshes().size());
    
    for (uint32_t mesh_idx = 0; mesh_idx < orig_meshes.size(); mesh_idx++) {
        const auto &orig_mesh = orig_meshes[mesh_idx];
        auto mesh_ptr = reinterpret_cast<VertexMesh<VertexType> *>(
            orig_mesh.get());

        auto processed = processMesh<VertexType>(*mesh_ptr, meshlet_buffer);

        if (processed.has_value()) {
            mesh_id_remap[mesh_idx] = processed_meshes.size();

            processed_meshes.emplace_back(move(*processed));
        } else {
            mesh_id_remap[mesh_idx] = ~0U;
        }
    }

    assert(processed_meshes.size() > 0);

    uint32_t num_vertices = 0;
    uint32_t num_indices = 0;
    uint32_t num_meshlets = 0;
    uint32_t num_chunks = 0;

    vector<MeshInfo> mesh_infos;
    for (auto &mesh : processed_meshes) {
        // Need to change all chunk offsets to be global to the whole scene
        for (auto &chunk : mesh.chunks) {
            chunk.indexOffset += num_indices;
            chunk.meshletOffset += num_meshlets;
        }

        // Rewrite indices to refer to the global vertex array
        // (Note this only really matters for RT to allow gl_CustomIndexEXT
        // to simply hold the base index of a mesh)
        for (uint32_t &idx : mesh.indices) {
            idx += num_vertices;
        }

        mesh_infos.push_back(MeshInfo {
            num_indices,
            num_chunks,
            uint32_t(mesh.indices.size() / 3),
            uint32_t(mesh.vertices.size()),
            uint32_t(mesh.chunks.size()),
        });

        num_vertices += mesh.vertices.size();
        num_indices += mesh.indices.size();
        num_meshlets += mesh.meshlets.size();
        num_chunks += mesh.chunks.size();
    }

    return ProcessedGeometry<VertexType> {
        move(processed_meshes),
        move(meshlet_buffer),
        move(mesh_id_remap),
        move(mesh_infos),
        num_vertices,
        num_indices,
        num_meshlets,
        num_chunks,
    };
}

// FIXME, needs to be rewritten with some way to maintain
// texture names from loading phase
static pair<vector<uint8_t>, MaterialMetadata> stageMaterials(
        const vector<shared_ptr<Material>> &materials)
{
    vector<filesystem::path> textures;
    unordered_map<string, size_t> texture_tracker;
    vector<size_t> param_offsets;
    param_offsets.reserve(materials.size());

    uint64_t num_material_bytes = 0;
    for (const auto &material : materials) {
        num_material_bytes += material->paramBytes.size();
    }

    vector<uint8_t> packed_params(num_material_bytes);
    uint8_t *cur_param_ptr = packed_params.data();

    uint32_t textures_per_material = 0;
    if (materials.size() > 0) {
        textures_per_material = materials[0]->textures.size();
    }

    vector<uint32_t> texture_indices;
    texture_indices.reserve(materials.size() * textures_per_material);

    for (const auto &material : materials) {
        memcpy(cur_param_ptr, material->paramBytes.data(),
               material->paramBytes.size());

        for (const auto &texture : material->textures) {
            // FIXME
            (void)texture;
            string texture_name = to_string(textures.size()) + ".ktx2";
            auto [iter, inserted] =
                texture_tracker.emplace(texture_name, textures.size());

            if (inserted) {
                textures.emplace_back(texture_name);
            }

            texture_indices.push_back(iter->second);
        }

        param_offsets.push_back(cur_param_ptr - packed_params.data());
        cur_param_ptr += material->paramBytes.size();
    }

    return {move(packed_params),
            {
                textures,
                uint32_t(materials.size()),
                textures_per_material,
                texture_indices,
            }};
}

void ScenePreprocessor::dump(string_view out_path_name)
{
    const SceneDescription &depth_desc = scene_data_->depthDesc;
    const SceneDescription &rgb_desc = scene_data_->rgbDesc;

    auto depth_geometry = processGeometry<DepthPipeline::Vertex>(depth_desc);
    auto rgb_geometry = processGeometry<RGBPipeline::Vertex>(rgb_desc);

    filesystem::path out_path(out_path_name);
    string basename = out_path.filename();
    basename.resize(basename.rfind('.'));

    ofstream out(out_path, ios::binary);
    auto write = [&](auto val) {
        out.write(reinterpret_cast<const char *>(&val), sizeof(decltype(val)));
    };

    // Pad to 256 (maximum uniform / storage buffer alignment requirement)
    auto write_pad = [&](size_t align_req = 256) {
        static char pad_buffer[64] = { 0 };
        size_t cur_bytes = out.tellp();
        size_t align = cur_bytes % align_req;
        if (align != 0) {
            out.write(pad_buffer, align_req - align);
        }
    };

    auto align_offset = [](size_t offset) {
        return (offset + 255) & ~255;
    };

    auto make_staging_header = [&](const auto &geometry,
                                   const vector<uint8_t> &material_params) {

        constexpr uint64_t vertex_size =
            sizeof(typename decltype(geometry.meshes[0].vertices)::value_type);
        uint64_t vertex_bytes = vertex_size * geometry.totalVertices;

        StagingHeader hdr;
        hdr.numVertices = geometry.totalVertices;
        hdr.numIndices = geometry.totalIndices;

        hdr.indexOffset = align_offset(vertex_bytes);

        hdr.meshletBufferOffset = align_offset(
            hdr.indexOffset + sizeof(uint32_t) * geometry.totalIndices);
        hdr.meshletBufferBytes =
            geometry.meshletBuffer.size() * sizeof(uint32_t);

        hdr.meshletOffset = align_offset(
                hdr.meshletBufferOffset + hdr.meshletBufferBytes);
        hdr.meshletBytes = geometry.totalMeshlets * sizeof(Meshlet);

        hdr.meshChunkOffset = align_offset(
                hdr.meshletOffset + hdr.meshletBytes);
        hdr.meshChunkBytes = geometry.totalChunks * sizeof(MeshChunk);

        hdr.materialOffset = align_offset(
                hdr.meshChunkOffset + hdr.meshChunkBytes);
        hdr.materialBytes = material_params.size();

        hdr.totalBytes = hdr.materialOffset + hdr.materialBytes;
        hdr.numMeshes = geometry.meshInfos.size();

        return hdr;
    };

    auto write_staging = [&](const auto &geometry,
                             const vector<uint8_t> &material_params,
                             const StagingHeader &hdr) {
        write_pad(256);

        auto stage_beginning = out.tellp();
        // Write all vertices
        for (auto &mesh : geometry.meshes) {
            constexpr uint64_t vertex_size =
                sizeof(typename decltype(mesh.vertices)::value_type);
            out.write(reinterpret_cast<const char *>(mesh.vertices.data()),
                      vertex_size * mesh.vertices.size());
        }

        write_pad(256);
        // Write all indices
        for (auto &mesh : geometry.meshes) {
            out.write(reinterpret_cast<const char *>(mesh.indices.data()),
                      mesh.indices.size() * sizeof(uint32_t));
        }

        write_pad(256);
        // Write meshlet buffer
        out.write(reinterpret_cast<const char *>(
                      geometry.meshletBuffer.data()),
                  hdr.meshletBufferBytes);

        write_pad(256);
        // Write meshlets
        for (auto &mesh : geometry.meshes) {
            out.write(reinterpret_cast<const char *>(mesh.meshlets.data()),
                      mesh.meshlets.size() * sizeof(Meshlet));
        }

        write_pad(256);
        // Write chunks
        for (auto &mesh : geometry.meshes) {
            out.write(reinterpret_cast<const char *>(mesh.chunks.data()),
                      mesh.chunks.size() * sizeof(MeshChunk));
        }

        write_pad(256);
        out.write(reinterpret_cast<const char *>(material_params.data()),
                  hdr.materialBytes);

        assert(out.tellp() == int64_t(hdr.totalBytes + stage_beginning));
    };

    // FIXME material system needs to be redone. Don't actually know
    // the texture names at this point.
    auto write_materials = [&](const MaterialMetadata &metadata) {
        write(uint32_t(metadata.textures.size()));
        for (uint32_t tex_idx = 0; tex_idx < metadata.textures.size();
             tex_idx++) {
            const string &tex_name =
                basename + "_" + to_string(tex_idx) + ".ktx2";
            out.write(tex_name.data(), tex_name.size());
            out.put(0);
        }

        write(uint32_t(metadata.numMaterials));
        write(uint32_t(metadata.texturesPerMaterial));
        out.write(reinterpret_cast<const char *>(
                      metadata.textureIndices.data()),
                  metadata.textureIndices.size() * sizeof(uint32_t));

    };

    auto write_instances = [&](const SceneDescription &desc,
                               const vector<uint32_t> &mesh_id_remap) {
        const vector<InstanceProperties> &instances =
            desc.getDefaultInstances();
        uint32_t num_instances = instances.size();
        for (const InstanceProperties &orig_inst : instances) {
            if (mesh_id_remap[orig_inst.meshIndex] == ~0U) {
                num_instances--;
            }
        }

        write(uint32_t(num_instances));
        for (const InstanceProperties &orig_inst : instances) {
            uint32_t new_mesh_id = mesh_id_remap[orig_inst.meshIndex];
            if (new_mesh_id == ~0U) continue;

            write(uint32_t(new_mesh_id));
            write(uint32_t(orig_inst.materialIndex));
            write(orig_inst.txfm);
        }
    };

    auto write_scene = [&](const auto &geometry,
                           const SceneDescription &desc) {
        const auto &materials = desc.getMaterials();
        auto [material_params, material_metadata] = stageMaterials(materials);

        StagingHeader hdr = make_staging_header(geometry, material_params);
        write(hdr);

        write_staging(geometry, material_params, hdr);

        // Write mesh infos
        out.write(reinterpret_cast<const char *>(geometry.meshInfos.data()),
                  hdr.numMeshes * sizeof(MeshInfo));

        write_materials(material_metadata);

        write_instances(desc, geometry.meshIDRemap);
        write_pad();
    };

    // FIXME should have more rigorous padding everywhere to make
    // mmap possible
    // Header: magic + depth offset + rgb offset (bytes from header)
    write(uint32_t(0x55555555));
    write(uint32_t(0));
    write(uint32_t(0)); // Rewrite later

    write_scene(depth_geometry, depth_desc);

    uint32_t rgb_offset = out.tellp();
    rgb_offset -= 12; // Account for 12 byte global header
    out.seekp(8);
    write(uint32_t(rgb_offset / sizeof(uint32_t)));
    out.seekp(rgb_offset, ios::cur);

    write_scene(rgb_geometry, rgb_desc);

    out.close();
}

template struct HandleDeleter<SceneData>;

}
