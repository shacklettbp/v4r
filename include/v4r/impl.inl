namespace v4r {

const glm::mat4 & CommandStream::getInstanceTransform(uint32_t batch_idx,
                                                      uint32_t inst_idx) const
{
    return cur_inputs_[batch_idx].instanceTransforms[inst_idx];
}

void CommandStream::updateInstanceTransform(uint32_t batch_idx,
                                            uint32_t inst_idx,
                                            const glm::mat4 &mat)
{
    cur_inputs_[batch_idx].instanceTransforms[inst_idx] = mat;
    cur_inputs_[batch_idx].dirty = true;
}

uint32_t CommandStream::numInstanceTransforms(uint32_t batch_idx) const
{
    return cur_inputs_[batch_idx].numInstances;
}

void CommandStream::setCameraView(uint32_t batch_idx, const glm::vec3 &eye,
                                  const glm::vec3 &look, const glm::vec3 &up)
{
    setCameraView(batch_idx, glm::lookAt(eye, look, up));
}

void CommandStream::setCameraView(uint32_t batch_idx, const glm::mat4 &m)
{
    *(cur_inputs_[batch_idx].view) = m;
}

void CommandStream::rotateCamera(uint32_t batch_idx, float angle,
                                 const glm::vec3 &axis)
{
    glm::mat4 &cur_view = *(cur_inputs_[batch_idx].view);
    cur_view = glm::rotate(cur_view, angle, axis);
}

void CommandStream::translateCamera(uint32_t batch_idx, const glm::vec3 &v)
{
    glm::mat4 &cur_view = *(cur_inputs_[batch_idx].view);
    cur_view = glm::translate(cur_view, v);
}

}
