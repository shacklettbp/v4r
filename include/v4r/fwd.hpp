#ifndef V4R_FWD_HPP_INCLUDED
#define V4R_FWD_HPP_INCLUDED

namespace v4r {

class VulkanState;
class CommandStreamState;

class LoaderState;

template <typename VertexType>
class Mesh;

template <typename DescriptionType>
class Material;

class Texture;

template <typename PipelineType>
class SceneDescription;

class Scene;
class EnvironmentState;
class Environment;

class CudaStreamState;
class CudaState;

template <typename PipelineType>
class CommandStream;

template <typename PipelineType>
class BatchRenderer;


}

#endif
