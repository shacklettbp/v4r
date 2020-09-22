#ifndef SCENE_HPP_INCLUDED
#define SCENE_HPP_INCLUDED

#include <v4r/config.hpp>
#include <v4r/assets.hpp>

#include <list>
#include <mutex>
#include <string_view>
#include <unordered_map>

#include "descriptors.hpp"
#include "utils.hpp"
#include "vulkan_handles.hpp"
#include "vulkan_memory.hpp"

// Forward declare ktxTexture as kind of an opaque backing data type
struct ktxTexture;

namespace v4r {

struct PerSceneDescriptorConfig {
    VkSampler textureSampler;
    VkDescriptorSetLayout layout;
};

struct Mesh {
    std::vector<uint32_t> indices;
};

template <typename VertexType>
struct VertexMesh : public Mesh {
    std::vector<VertexType> vertices;

    VertexMesh(std::vector<VertexType> vertices,
               std::vector<uint32_t> indices);
};

template <typename VertexType>
struct VertexImpl;

struct Texture {
    uint32_t width;
    uint32_t height;
    uint32_t numLevels;

    ktxTexture *data;

    ~Texture();
};

namespace MaterialParam {
    struct DiffuseColorTexture {
        std::shared_ptr<Texture> value;
    };

    struct DiffuseColorUniform {
        glm::vec4 value;
    };

    using AlbedoColorTexture = DiffuseColorTexture;
    using AlbedoColorUniform = DiffuseColorUniform;

    struct SpecularColorTexture {
        std::shared_ptr<Texture> value;
    };

    struct SpecularColorUniform {
        glm::vec4 value;
    };

    struct ShininessUniform {
        float value;
    };
};

struct Material {
    std::vector<std::shared_ptr<Texture>> textures;
    std::vector<uint8_t> paramBytes;
};

template <typename MaterialParamsType>
struct MaterialImpl;

struct EnvironmentInit {
    EnvironmentInit(
            const std::vector<InstanceProperties>
                &instances,
            const std::vector<LightProperties> &lights,
            uint32_t num_meshes);

    std::vector<std::vector<glm::mat4x3>> transforms;
    std::vector<std::vector<uint32_t>> materials;
    std::vector<std::pair<uint32_t, uint32_t>> indexMap;
    std::vector<std::vector<uint32_t>> reverseIDMap;

    std::vector<LightProperties> lights;
    std::vector<uint32_t> lightIDs;
    std::vector<uint32_t> lightReverseIDs;
};

struct Scene {
    std::vector<LocalImage> textures;
    std::vector<VkImageView> texture_views;
    DescriptorSet materialSet;
    DescriptorSet cullSet;
    LocalBuffer data;
    VkDeviceSize indexOffset;
    uint32_t numMeshes;
    EnvironmentInit envDefaults;
};

class EnvironmentState {
public:
    EnvironmentState(const std::shared_ptr<Scene> &s, const glm::mat4 &proj);

    std::shared_ptr<Scene> scene;
    glm::mat4 projection;

    std::vector<std::vector<uint32_t>> reverseIDMap;
    std::vector<uint32_t> freeIDs;

    std::vector<LightProperties> lights;
    std::vector<uint32_t> freeLightIDs;
    std::vector<uint32_t> lightIDs;
    std::vector<uint32_t> lightReverseIDs;
};

struct MaterialMetadata {
    std::vector<std::shared_ptr<Texture>> textures;
    uint32_t numMaterials;
    uint32_t texturesPerMaterial;
    std::vector<uint32_t> textureIndices;
};

struct StagedScene {
    HostBuffer buffer;
    VkDeviceSize indexBufferOffset;
    VkDeviceSize paramBufferOffset;
    VkDeviceSize numMaterialParamBytes;
    VkDeviceSize meshInfoOffset;
    uint32_t numMeshes;
    VkDeviceSize totalBytes;
};

struct SceneLoadInfo {
    StagedScene scene;
    MaterialMetadata materialMetadata;
    EnvironmentInit envInit;
};

struct LoaderImpl {
    std::add_pointer_t<
        StagedScene(const std::vector<std::shared_ptr<Mesh>> &,
                    const std::vector<uint8_t> &param_bytes,
                    const DeviceState &,
                    MemoryAllocator &)>
            stageScene;

    std::add_pointer_t<
        SceneDescription(std::string_view, const glm::mat4 &)>
            parseScene;

    std::add_pointer_t<
        std::shared_ptr<Mesh>(std::string_view)>
            loadMesh;

    template <typename VertexType, typename MaterialParamsType>
    static LoaderImpl create();
};

class LoaderState {
public:
    LoaderState(const DeviceState &dev,
                const LoaderImpl &impl,
                const VkDescriptorSetLayout &scene_set_layout,
                DescriptorManager::MakePoolType make_scene_pool,
                const VkDescriptorSetLayout &mesh_cull_scene_set_layout,
                DescriptorManager::MakePoolType make_mesh_cull_scene_pool,
                MemoryAllocator &alc,
                QueueManager &queue_manager,
                const glm::mat4 &coordinateTransform);


    std::shared_ptr<Scene> loadScene(std::string_view scene_path);

    std::shared_ptr<Scene> makeScene(
            const SceneDescription &desc);

    std::shared_ptr<Texture> loadTexture(
            const std::vector<uint8_t> &raw);

    template <typename MaterialParamsType>
    std::shared_ptr<Material> makeMaterial(MaterialParamsType params);

    std::shared_ptr<Mesh> loadMesh(std::string_view geometry_path);

    template <typename VertexType>
    std::shared_ptr<Mesh> makeMesh(std::vector<VertexType> vertices,
                                   std::vector<uint32_t> indices);
                
    const DeviceState &dev;

    const VkCommandPool gfxPool;
    const QueueState &gfxQueue;
    const VkCommandBuffer gfxCopyCommand;

    const VkCommandPool transferPool;
    const QueueState &transferQueue;
    const VkCommandBuffer transferStageCommand;

    const VkSemaphore semaphore;
    const VkFence fence;

    MemoryAllocator &alloc;
    DescriptorManager descriptorManager;
    DescriptorManager cullDescriptorManager;

    glm::mat4 coordinateTransform;

private:
    std::shared_ptr<Scene> makeScene(
            SceneLoadInfo load_info);

    const LoaderImpl impl_;
};

}

#endif
