#ifndef V4R_HPP_INCLUDED
#define V4R_HPP_INCLUDED

#include <v4r/assets.hpp>
#include <v4r/config.hpp>
#include <v4r/environment.hpp>
#include <v4r/fwd.hpp>
#include <v4r/stats.hpp>
#include <v4r/utils.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string_view>
#include <vector>

namespace v4r {

class AssetLoader {
public:
    std::shared_ptr<Mesh> loadMesh(std::string_view geometry_path);

    template <typename VertexType>
    std::shared_ptr<Mesh> loadMesh(
            std::vector<VertexType> vertices,
            std::vector<uint32_t> indices);

    std::shared_ptr<Texture> loadTexture(std::string_view texture_path);

    template <typename MaterialParamsType>
    std::shared_ptr<Material> makeMaterial(
            MaterialParamsType params);

    std::shared_ptr<Scene> makeScene(
            const SceneDescription &desc);

    // Shortcut for Gibson style scene files
    std::shared_ptr<Scene> loadScene(std::string_view scene_path);

private:
    AssetLoader(Handle<LoaderState> &&state);

    Handle<LoaderState> state_;

friend class BatchRenderer;
};

class CommandStream {
public:
    Environment makeEnvironment(const std::shared_ptr<Scene> &scene,
                                float hfov, float near = 0.001f,
                                float far = 10000.f);
    // Render batch
    uint32_t render(const std::vector<Environment> &elems);

    void waitForFrame(uint32_t frame_id = 0);

protected:
    CommandStream(Handle<CommandStreamState> &&state,
                  uint32_t render_width,
                  uint32_t render_height);

    Handle<CommandStreamState> state_;

private:
    uint32_t render_width_;
    uint32_t render_height_;

friend class BatchRenderer;
};

class BatchRenderer {
public:
    template <typename PipelineType>
    BatchRenderer(const RenderConfig &cfg,
                  const RenderFeatures<PipelineType> &features);

    AssetLoader makeLoader();
    CommandStream makeCommandStream();

    Statistics getStatistics() const;

protected:
    BatchRenderer(Handle<VulkanState> &&vk_state);
    Handle<VulkanState> state_;
};

}

#endif
