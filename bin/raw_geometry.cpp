#include <v4r.hpp>
#include <v4r/debug.hpp>

#include <iostream>
#include <cstdlib>
#include <glm/gtc/matrix_transform.hpp>

using namespace std;
using namespace v4r;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Specify scene" << endl;
        exit(EXIT_FAILURE);
    }

    BatchRenderer renderer({0, 1, 1, 1, 256, 256,
        glm::mat4(1.f), // global transform is just identity
        {
            RenderFeatures::MeshColor::Texture,
            RenderFeatures::Pipeline::Unlit,
            RenderFeatures::Outputs::Color |
                RenderFeatures::Outputs::Depth,
            RenderFeatures::Options::CpuSynchronization
        }
    });

    AssetLoader loader = renderer.makeLoader();
    CommandStream cmd_stream = renderer.makeCommandStream();

    // Renderdoc entry points (ignore)
    RenderDoc rdoc {};
    rdoc.startFrame();

    vector<shared_ptr<Mesh>> meshes;
    vector<shared_ptr<Material>> materials;

    using Vertex = UnlitRendererInputs::TexturedVertex;
    using MaterialDescription = UnlitRendererInputs::MaterialDescription;

    // This block populates the above vectors with meshes
    // and a material
    {
        // Construct geometry
        vector<Vertex> single_tri_verts {
            Vertex {
                glm::vec3(0.f, 0.f, -1.5f),
                glm::vec2(0.f, 0.f)
            },
            Vertex {
                glm::vec3(0.f, 1.f, -1.5f),
                glm::vec2(0.f, 1.f)
            },
            Vertex {
                glm::vec3(1.f, 1.f, -1.5f),
                glm::vec2(1.f, 1.f)
            },
        };

        vector<uint32_t> single_tri_idxs {
            2, 1, 0
        };

        vector<Vertex> quad_verts {
            Vertex {
                glm::vec3(0.f, 0.f, -1.6f),
                glm::vec2(0.f, 0.f)
            },
            Vertex {
                glm::vec3(0.f, 1.f, -1.6f),
                glm::vec2(0.f, 1.f)
            },
            Vertex {
                glm::vec3(1.f, 1.f, -1.6f),
                glm::vec2(1.f, 1.f)
            },
            Vertex {
                glm::vec3(1.f, 0.f, -1.6f),
                glm::vec2(1.f, 0.f)
            }
        };

        vector<uint32_t> quad_idxs {
            2, 1, 0,
            3, 2, 0
        };

        // Load these vertex / index buffers into the renderer
        meshes.emplace_back(loader.loadMesh(move(single_tri_verts),
                                            move(single_tri_idxs)));

        meshes.emplace_back(loader.loadMesh(move(quad_verts),
                                            move(quad_idxs)));

        // Load texture off disk
        shared_ptr<Texture> texture = loader.loadTexture(argv[1]);

        materials.emplace_back(loader.makeMaterial(
                    MaterialDescription { texture }));
    }

    // Get a opaque handle to the scene (collection of meshes and materials).
    // SceneDescription retains ownership of meshes and materials vector
    // after makeScene is called (in case similar scenes need to be created),
    // so this block drops the description after the call to makeScene.
    shared_ptr<Scene> scene;
    {
        SceneDescription scene_desc(move(meshes), move(materials));
        // Add the triangle (mesh_idx 0) as a default instance in all
        // environments derived from this scene
        scene_desc.addInstance(0, 0, glm::mat4(1.f));

        scene = loader.makeScene(scene_desc);
    }

    // Batch size of 1, so this vector has one element in this example
    vector<Environment> envs;
    envs.emplace_back(cmd_stream.makeEnvironment(scene, 90, 0.01, 1000));
    envs[0].setCameraView(glm::mat4(1.f)); // Set view matrix to identity

    // Render the environment (unchanged from default)
    auto sync = cmd_stream.render(envs);
    sync.cpuWait();
    saveFrame("/tmp/out_0.bmp", cmd_stream.getColorDevPtr(), 256, 256, 4);

    // Now add in an instance of the quad mesh
    uint32_t inst_id = envs[0].addInstance(1, 0, glm::mat4(1.f));

    sync = cmd_stream.render(envs);
    sync.cpuWait();
    saveFrame("/tmp/out_1.bmp", cmd_stream.getColorDevPtr(), 256, 256, 4);

    // Finally, move the quad mesh instance
    envs[0].updateInstanceTransform(inst_id,
                                    glm::translate(
                                         envs[0].getInstanceTransform(inst_id),
                                         glm::vec3(0.5f, 0.f, 0.f)));

    sync = cmd_stream.render(envs);
    sync.cpuWait();
    saveFrame("/tmp/out_2.bmp", cmd_stream.getColorDevPtr(), 256, 256, 4);

    rdoc.endFrame();
}
