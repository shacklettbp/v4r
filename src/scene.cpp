#include "scene.hpp"

#include "utils.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/gtc/type_ptr.hpp>

#include <cassert>
#include <iostream>

using namespace std;

namespace v4r {

static SceneAssets loadAssets(const string &scene_path)
{
    Assimp::Importer importer;
    int flags = aiProcess_PreTransformVertices | aiProcess_Triangulate;
    const aiScene *raw_scene = importer.ReadFile(scene_path.c_str(), flags);
    if (!raw_scene) {
        cerr << "Failed to load scene " << scene_path << ": " <<
            importer.GetErrorString() << endl;
        fatalExit();
    }

    vector<Vertex> vertices;
    vector<uint32_t> indices;
    vector<SceneMesh> meshes;

    uint32_t cur_vertex_index = 0;

    for (uint32_t mesh_idx = 0; mesh_idx < raw_scene->mNumMeshes; mesh_idx++) {
        aiMesh *raw_mesh = raw_scene->mMeshes[mesh_idx];

        bool has_uv = raw_mesh->HasTextureCoords(0);
        bool has_color = raw_mesh->HasVertexColors(0);

        for (uint32_t vert_idx = 0; vert_idx < raw_mesh->mNumVertices; vert_idx++) {
            glm::vec3 pos = glm::make_vec3(&raw_mesh->mVertices[vert_idx].x);
            pos.y = -pos.y;

            Vertex vertex {
                pos,
                has_uv ?
                    glm::make_vec2(&raw_mesh->mTextureCoords[0][vert_idx].x) :
                    glm::vec2(0.f),
                has_color ?
                    glm::make_vec3(&raw_mesh->mColors[0][vert_idx].r) :
                    glm::vec3(1.0f),
            };

            vertices.emplace_back(move(vertex));
        }

        for (uint32_t face_idx = 0; face_idx < raw_mesh->mNumFaces; face_idx++) {
            for (uint32_t tri_idx = 0; tri_idx < 3; tri_idx++) {
                indices.push_back(raw_mesh->mFaces[face_idx].mIndices[tri_idx]);
            }
        }

        meshes.emplace_back(SceneMesh {
            cur_vertex_index,
            raw_mesh->mNumFaces * 3
        });

        cur_vertex_index += raw_mesh->mNumFaces * 3;
    }

    return SceneAssets {
        move(vertices),
        move(indices),
        move(meshes)
    };
}

Scene::Scene(const std::string &scene_path, const VulkanState &render_state)
    : path_(scene_path),
      ref_count_(1),
      state_(render_state.loadScene(loadAssets(scene_path)))
{}

SceneManager::SceneManager(const VulkanState &renderer_state)
    : renderer_state_(renderer_state),
      load_mutex_(),
      scenes_(),
      scene_lookup_()
{}

SceneID SceneManager::loadScene(const std::string &scene_path)
{
    scoped_lock lock(load_mutex_);

    auto lookup_iter = scene_lookup_.find(scene_path);
    if (lookup_iter != scene_lookup_.end()) {
        SceneID id = lookup_iter->second;
        id.iter_->refIncrement();

        return id;
    } 

    scenes_.emplace_front(scene_path, renderer_state_);
    auto scene_iter = scenes_.begin();

    SceneID id { scene_iter };

    scene_lookup_.emplace(scene_path, id);

    return id;
}

void SceneManager::dropScene(SceneID &&scene_id)
{
    scoped_lock lock(load_mutex_);
    
    bool should_free = scene_id.iter_->refDecrement();
    if (!should_free) return;

    [[maybe_unused]] size_t num_erased =
        scene_lookup_.erase(scene_id.iter_->getPath());

    assert(num_erased == 1);

    scenes_.erase(scene_id.iter_);
}

}
