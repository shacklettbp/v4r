#include "scene.hpp"

#include "utils.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION

#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace std;

namespace v4r {

static const Texture * loadTexture(const aiScene *raw_scene,
        const aiMaterial *raw_mat,
        aiTextureType type,
        list<Texture> &textures,
        unordered_map<string, const Texture *> &loaded_texture_lookup)
{
    aiString tex_path;
    bool has_texture = raw_mat->Get(AI_MATKEY_TEXTURE(type, 0), tex_path) ==
        AI_SUCCESS;

    if (!has_texture) return nullptr;

    auto lookup = loaded_texture_lookup.find(tex_path.C_Str());

    if (lookup != loaded_texture_lookup.end()) return lookup->second;

    if (auto texture = raw_scene->GetEmbeddedTexture(tex_path.C_Str())) {
        if (texture->mHeight > 0) {
            cerr << "Uncompressed textures not supported" << endl;
            fatalExit();
        } else {
            const uint8_t *raw_input =
                reinterpret_cast<const uint8_t *>(texture->pcData);
            if (stbi_is_hdr_from_memory(raw_input, texture->mWidth)) {
                cerr << "HDR textures not supported" << endl;
                fatalExit();
            }

            int width, height, num_channels;
            uint8_t *texture_data =
                stbi_load_from_memory(raw_input, texture->mWidth,
                                      &width, &height, &num_channels, 3);

            if (texture_data == nullptr) {
                cerr << "Failed to load texture" << endl;
                fatalExit();
            }

            if (num_channels != 3) {
                cerr << "Only RGB888 textures supported" << endl;
                fatalExit();
            }

            Texture &new_tex = textures.emplace_back(Texture {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height),
                3,
                ManagedArray<uint8_t>(texture_data, stbi_image_free)
            });

            loaded_texture_lookup.emplace(tex_path.C_Str(), &new_tex);

            return &new_tex;
        }
    } else {
        // FIXME
        cerr << "External textures not supported yet" << endl;
        fatalExit();
    }
}

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

    list<Texture> textures;
    vector<Material> materials;
    
    unordered_map<string, const Texture *> loaded_texture_lookup;

    for (uint32_t mat_idx = 0; mat_idx < raw_scene->mNumMaterials; mat_idx++) {
        const aiMaterial *raw_mat = raw_scene->mMaterials[mat_idx];

        auto ambient_tex = loadTexture(raw_scene, raw_mat,
                                       aiTextureType_AMBIENT,
                                       textures,
                                       loaded_texture_lookup);
        glm::vec4 ambient_color {};
        if (!ambient_tex) {
            aiColor4D color;
            raw_mat->Get(AI_MATKEY_COLOR_AMBIENT, color);
            ambient_color = glm::vec4(color.r, color.g, color.b, color.a);
        }

        auto diffuse_tex = loadTexture(raw_scene, raw_mat,
                                       aiTextureType_DIFFUSE,
                                       textures,
                                       loaded_texture_lookup);
        glm::vec4 diffuse_color {};
        if (!diffuse_tex) {
            aiColor4D color;
            raw_mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
            diffuse_color = glm::vec4(color.r, color.g, color.b, color.a);
        }

        auto specular_tex = loadTexture(raw_scene, raw_mat,
                                        aiTextureType_SPECULAR,
                                        textures,
                                        loaded_texture_lookup);
        glm::vec4 specular_color {};
        if (!specular_tex) {
            aiColor4D color;
            raw_mat->Get(AI_MATKEY_COLOR_SPECULAR, color);
            specular_color = glm::vec4(color.r, color.g, color.b, color.a);
        }

        materials.emplace_back(Material {
            ambient_tex,
            ambient_color,
            diffuse_tex,
            diffuse_color,
            specular_tex,
            specular_color
        });
    }

    vector<Vertex> vertices;
    vector<uint32_t> indices;
    vector<SceneMesh> meshes;

    uint32_t cur_vertex_index = 0;

    for (uint32_t mesh_idx = 0; mesh_idx < raw_scene->mNumMeshes; mesh_idx++) {
        aiMesh *raw_mesh = raw_scene->mMeshes[mesh_idx];

        unsigned int mat_idx = raw_mesh->mMaterialIndex;
        cout << "Material: " << mat_idx << endl;

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
        move(textures),
        move(materials),
        move(vertices),
        move(indices),
        move(meshes)
    };
}

Scene::Scene(const std::string &scene_path, CommandStreamState &renderer_state)
    : path_(scene_path),
      ref_count_(1),
      state_(renderer_state.loadScene(loadAssets(scene_path)))
{}

SceneManager::SceneManager()
    : load_mutex_(),
      scenes_(),
      scene_lookup_()
{}

SceneID SceneManager::loadScene(const std::string &scene_path,
                                CommandStreamState &renderer_state)
{
    scoped_lock lock(load_mutex_);

    auto lookup_iter = scene_lookup_.find(scene_path);
    if (lookup_iter != scene_lookup_.end()) {
        SceneID id = lookup_iter->second;
        id.iter_->refIncrement();

        return id;
    } 

    scenes_.emplace_front(scene_path, renderer_state);
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
