namespace v4r {

InstanceProperties::InstanceProperties(const glm::mat4 &model_txfm, uint32_t mat_idx)
    : modelTransform(model_txfm),
      materialIndex(mat_idx)
{}

template <typename PipelineType>
SceneDescription<PipelineType>::SceneDescription(
        std::vector<std::shared_ptr<MeshType>> meshes,
        std::vector<std::shared_ptr<MaterialType>> materials)
    : meshes_(move(meshes)),
      materials_(move(materials)),
      default_instances_(meshes_.size())
{}

template <typename PipelineType>
void SceneDescription<PipelineType>::addInstance(uint32_t model_idx,
        uint32_t material_idx,
        const glm::mat4 &model_transform)
{
    default_instances_[model_idx].emplace_back(model_transform, material_idx);
}

template <typename PipelineType>
const std::vector<std::shared_ptr<Mesh<typename PipelineType::VertexType>>> &
SceneDescription<PipelineType>::getMeshes() const
{
    return meshes_;
}

template <typename PipelineType>
const std::vector<std::shared_ptr<Material<typename
        PipelineType::MaterialDescType>>> &
SceneDescription<PipelineType>::getMaterials() const
{
    return materials_;
}

template <typename PipelineType>
const std::vector<std::vector<InstanceProperties>> &
SceneDescription<PipelineType>::getDefaultInstances() const
{
    return default_instances_;
}

}
