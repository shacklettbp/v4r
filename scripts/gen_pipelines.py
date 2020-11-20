#!/usr/bin/env python

import sys
import os
import copy
import itertools
import json
from collections import defaultdict, namedtuple

def get_combos(arr):
    combos = []
    for i in range(1, len(arr) + 1):
        i_combos = itertools.combinations(arr, i)
        combos += list(i_combos)

    return combos

class InterfaceTracker:
    def __init__(self):
        self.cur_vert_in_idx = 0
        self.cur_vert_out_idx = 0
        self.cur_frag_out_idx = 0
        self.cur_binding_idx = defaultdict(lambda: 0)

    def vert_in(self):
        cur = self.cur_vert_in_idx
        self.cur_vert_in_idx += 1
        return cur

    def vert_out(self):
        cur = self.cur_vert_out_idx
        self.cur_vert_out_idx += 1
        return cur

    def frag_out(self):
        cur = self.cur_frag_out_idx
        self.cur_frag_out_idx += 1
        return cur

    def bind(self, set_idx):
        cur = self.cur_binding_idx[set_idx]
        self.cur_binding_idx[set_idx] += 1
        return cur

def add_default_definitions(param_definitions):
    param_definitions["bool"] = {
        "values": {
            True: [],
            False: []
        },
        "multiple": False
    }

    return param_definitions

def generate_combinations(valid_configs, params, param_definitions):
    for valid in valid_configs:
        def get_allowed_config(param_iter):
            try:
                param_name = next(param_iter)
            except StopIteration:
                yield {}
                return

            param_type = params[param_name]["type"]
            allowed_values = valid['allowed'].get(param_name) or \
                    param_definitions[param_type]['values'].keys()

            required = valid['required'].get(param_name) or []
            
            multival = param_definitions[param_type]['multiple']

            def requirement_satisfied(val):
                for req_val in required:
                    if multival:
                        if not req_val in val:
                            return False
                    else:
                        if val != req_val:
                            return False

                return True

            if multival:
                allowed_values = get_combos(allowed_values)

            for possible_value in allowed_values:
                if not requirement_satisfied(possible_value):
                    continue

                subconfig_combos = get_allowed_config(copy.copy(param_iter))
                for subconfig in subconfig_combos:
                    if multival:
                        config = { param_name: possible_value }
                    else:
                        config = { param_name: (possible_value,) }

                    config.update(subconfig)

                    yield config

        allowed_configs = get_allowed_config(iter(params))
        for config in allowed_configs:
            yield config

VertexAttribute = namedtuple('VertexAttribute', ['name', 'type',
        'in_loc', 'out_loc'])

class Vertex:
    def __init__(self):
        self.attributes = []
        self.properties = []

    def add_attribute(self, name, type, in_loc, out_loc=None):
        self.attributes.append(VertexAttribute(name=name, type=type,
            in_loc=in_loc, out_loc=out_loc))

    def add_property(self, name, enable):
        self.properties.append((name, enable))

    def get_defines(self):
        defines = []

        for attr in self.attributes:
            defines.append(f"{attr.name}_IN_LOC={attr.in_loc}".upper())
            if attr.out_loc != None:
                defines.append(f"{attr.name}_LOC={attr.out_loc}".upper())

        return defines

    def gen_struct(self):
        cpp_attrs = [ f"glm::{attr.type} {attr.name}" 
                for attr in self.attributes]

        sep = ";\n        "
        return \
f"""    struct Vertex {{
        {sep.join(cpp_attrs)};
    }};"""

    def gen_impl(self, parent_type):
        sep =";\n    "

        cpp_props = [ f"static constexpr bool {prop} = \
{'true' if enable else 'false'}" for prop, enable in self.properties ]

        return \
f"""template <>
struct VertexImpl<{parent_type}::Vertex> {{
    {sep.join(cpp_props)};
}};
"""

    def gen_attrs(self, pipeline_type):
        attrs = []

        for attr in self.attributes:
            if attr.type == "vec3":
                vk_fmt = "VK_FORMAT_R32G32B32_SFLOAT"
            elif attr.type == "u8vec3":
                vk_fmt = "VK_FORMAT_R8G8B8_UNORM"
            elif attr.type == "vec2":
                vk_fmt = "VK_FORMAT_R32G32_SFLOAT"
            else:
                raise Exception("Unknown vertex attribute type")

            attrs.append(
f"""{{ {len(attrs)}, 0, {vk_fmt},
          offsetof(VertexType, {attr.name}) }}""")

        sep = ",\n        "

        return \
f"""
    using VertexType = {pipeline_type}::Vertex;

    static constexpr std::array<VkVertexInputAttributeDescription,
                                {len(attrs)}> vertexAttributes {{{{
        {sep.join(attrs)}
    }}}};
"""

MaterialParam = namedtuple('MaterialParam',
        ['name', 'value_type', 'qual_type', 'attr_type', 'is_texture'])

def compute_bytes(type_str):
    if type_str == 'vec4':
        return 16
    elif type_str == 'vec3':
        return 12
    elif type_str == 'vec2':
        return 8
    elif type_str == 'float':
        return 4
    else:
        raise Exception(f"Don't know size of {type_str}")

class PackedMaterial:
    def __init__(self, params):
        params = [p for p in params if not p.is_texture]

        param_bytes = [compute_bytes(p.value_type) for p in params]
        self.total_bytes = sum(param_bytes)
        self.end_pad = 0
        self.locations = {}

        if self.total_bytes == 0:
            return

        if self.total_bytes % 16 != 0:
            self.end_pad = 16 - (self.total_bytes % 16)

        self.total_bytes += self.end_pad

        assert((self.total_bytes % 16) == 0)

        self.num_vecs = self.total_bytes // 16

        sized_params = { 1: [], 2: [], 3: [], 4: [] }

        for param, num_bytes in zip(params, param_bytes):
            assert(num_bytes % 4 == 0)
            sized_params[num_bytes // 4].append(param)

        cur_offset = 0
        cur_vec4 = 0
        cur_bytes = 0
        while cur_bytes + self.end_pad < self.total_bytes:
            remaining = 4 - cur_offset

            def param_in_range(min_components, max_components):
                for num_components in reversed(range(min_components,
                                                     max_components + 1)):
                    if len(sized_params[num_components]) > 0:
                        return (sized_params[num_components].pop(), 
                                num_components)

                return None

            optimal = param_in_range(1, remaining)
            if optimal != None:
                param, components_used = optimal
            else:
                param, components_used = param_in_range(2, 3)

            if components_used > remaining:
                loc = [(cur_vec4, cur_offset, remaining)]

                components_used -= remaining
                cur_vec4 += 1
                loc.append((cur_vec4, 0, components_used))
            else:
                loc = [(cur_vec4, cur_offset, components_used)]
            
            self.locations[param] = loc

            if remaining - components_used == 0:
                cur_vec4 += 1
                cur_offset = 0
            else:
                cur_offset += components_used

            cur_bytes = (cur_vec4 * 4 + cur_offset) * 4

    def num_param_vecs(self):
        return self.num_vecs

    def num_end_bytes(self):
        return self.end_pad

    def gen_access_flags(self):
        flags = []
        for param, loc in self.locations.items():
            total_components = sum([l[2] for l in loc])
            if total_components == 1:
                aggregate = "float"
            else:
                aggregate = f"vec{total_components}"

            swizzle = "xyzw"

            access_fragments = []
            for base, offset, size in loc:
                swizzle_fragment = swizzle[offset:offset+size]
                access_fragments.append(
                    f"params.data[{base}].{swizzle_fragment}")

            access_code = f"{aggregate}({', '.join(access_fragments)})"

            flags.append(f'{param.name.upper()}_ACCESS="\\"{access_code}\\""')

        return flags

class Material:
    def __init__(self):
        self.params = []
        self.num_textures = 0
        self.num_params = 0

    def add_param(self, name, value_type, qual_type, attr_type, is_texture):
        self.params.append(MaterialParam(name=name,
            value_type=value_type, qual_type=qual_type, attr_type=attr_type,
            is_texture=is_texture))

        if is_texture:
            self.num_textures += 1
        else:
            self.num_params += 1

    def gen_struct(self):
        members = [ f"{param.qual_type} {param.name}"
                for param in self.params]
        sep = ";\n        "
        return \
f"""    struct MaterialParams {{
        {sep.join(members)};
    }};"""


    def pack(self):
        return PackedMaterial(self.params)

    def gen_impl(self, parent_type, packed):
        texture_moves = []

        uniform_members = []
        uniform_init = []
        generic_params = []

        for name, _, qual_type, attr_type, is_texture in self.params:
            if is_texture: 
                texture_moves.append(f"std::move(params.{name})")

            generic_params.append(
                    f"std::move(std::get<typename MaterialParam::{attr_type}&>\
(tuple_args).value)")

        for param in packed.locations.keys():
            uniform_members.append(f"{param.qual_type} {param.name}")
            uniform_init.append(f"params.{param.name}")

        num_pad = packed.num_end_bytes()
        if num_pad > 0:
            uniform_members.append(f"char pad[{num_pad}] {{}}")

        generic_sep = ",\n            "
        texture_sep = ",\n                "
        texture_init = f"""{{
                {texture_sep.join(texture_moves)}
            }}"""

        if len(uniform_members) > 0:
            block_sep = ";\n            " 
            block_members = block_sep.join(uniform_members)
            block_init = ", ".join(uniform_init)
            param_block_init = \
f"""struct {{
            {block_members};
        }} uniform_block {{ {block_init} }};

        std::vector<uint8_t> param_block(sizeof uniform_block);

        memcpy(param_block.data(), &uniform_block, param_block.size());"""

        else:
            param_block_init = "std::vector<uint8_t> param_block(0);"

        return \
f"""template <>
struct MaterialImpl<{parent_type}::MaterialParams> {{
    static std::shared_ptr<Material> make(
            {parent_type}::MaterialParams params)
    {{
        {param_block_init}

        return std::shared_ptr<Material>(new Material {{
            {texture_init},
            move(param_block)
        }});
    }}

    template<typename... Args>
    static std::shared_ptr<Material> make(Args ...args)
    {{
        auto tuple_args = std::forward_as_tuple(args...);
        return make({parent_type}::MaterialParams {{
            {generic_sep.join(generic_params)}
        }});
    }}
}};
"""
    
    def gen_scene_layout(self):
        bindings = []
        if self.num_textures > 0:
            bindings.append(
f"""BindingConfig<{len(bindings)}, VK_DESCRIPTOR_TYPE_SAMPLER, 1,
                      VK_SHADER_STAGE_FRAGMENT_BIT>""")

            for i in range(self.num_textures):
                bindings.append(
f"""BindingConfig<{len(bindings)}, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                      VulkanConfig::max_materials,
                      VK_SHADER_STAGE_FRAGMENT_BIT,
                      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT>""")

        if self.num_params > 0:
            bindings.append(
f"""BindingConfig<{len(bindings)}, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                      VK_SHADER_STAGE_FRAGMENT_BIT>""")

        if len(bindings) == 0:
            return None

        sep = f",\n        "
        return \
f"""using PerSceneLayout = DescriptorLayout<
        {sep.join(bindings)}
    >;"""

def build_vertex(props, iface):
    vertex = Vertex()
    vertex.add_attribute("position", type="vec3", in_loc=iface.vert_in())
    vertex.add_property("hasPosition", True)

    if "needLighting" in props:
        vertex.add_attribute("normal", type="vec3", in_loc=iface.vert_in(),
            out_loc=iface.vert_out())
        vertex.add_property("hasNormal", True)
    else:
        vertex.add_property("hasNormal", False)

    if "needTextures" in props:
        vertex.add_attribute("uv", type="vec2", in_loc=iface.vert_in(),
            out_loc=iface.vert_out())
        vertex.add_property("hasUV", True)
    else:
        vertex.add_property("hasUV", False)

    if "needVertexColor" in props:
        vertex.add_attribute("color", type="u8vec3", in_loc=iface.vert_in(),
            out_loc=iface.vert_out())
        vertex.add_property("hasColor", True)
    else:
        vertex.add_property("hasColor", False)

    return vertex

def cpp_value(param_name, value, pipeline_params):
    param_type = pipeline_params[param_name]["type"]
    if param_type == "bool":
        return 'true' if value else 'false'
    else:
        return f"{param_type}::{value}"

def generate_pipeline_definition(name, params):
    cpp_params = []
    for param_name, param_details in params.items():
        param_str = f"{param_details['type']} {param_name}"
        default = param_details.get("default")
        if default != None:
            param_str += f" = {cpp_value(param_name, default, params)}"

        cpp_params.append(param_str)

    first_param = next(iter(params))

    return f"""template <{", ".join(cpp_params)}>
struct {name} {{
    static_assert({first_param} != {first_param},
                  "Unsupported combination of pipeline parameters");
}};
"""

def specialization_type(name, config, pipeline_params):
    param_values = ", ".join([" | ".join(
        [cpp_value(k, e, pipeline_params) for e in v]) 
            for k, v in config.items()])

    return f"{name}<{param_values}>"

def generate_pipeline_specialization(specialization_type,
        vertex_defn, material_defn):
    return f"""template <>
struct {specialization_type} {{
{vertex_defn}

{material_defn}
}};
"""

def generate_props_specialization(specialization_type, props,
        all_props, vertex, material, shader_name, txfm_loc, mat_loc):

    prop_members = []

    for prop in all_props:
        prop_val = 'true' if prop in props else 'false'
        prop_members.append(f"static constexpr bool {prop} = {prop_val}")

    sep = ";\n    "

    vert_input_locs = [
            f"static constexpr uint32_t transformLocationVertex = {txfm_loc}"]

    if mat_loc != None:
        vert_input_locs.append(
            f"static constexpr uint32_t materialLocationVertex = {mat_loc}")

    frame_bindings = []

    if "needLighting" in props:
        view_info_stages = """VK_SHADER_STAGE_VERTEX_BIT |
                      VK_SHADER_STAGE_FRAGMENT_BIT"""
                 
        frame_bindings.append(
"""BindingConfig<1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                      VK_SHADER_STAGE_FRAGMENT_BIT>""")
    else:
        view_info_stages = "VK_SHADER_STAGE_VERTEX_BIT"
        

    frame_bindings = [
f"""BindingConfig<0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                      {view_info_stages}>"""
            ] + frame_bindings

    frame_sep = ",\n        "

    layouts = \
f"""using PerFrameLayout = DescriptorLayout<
        {frame_sep.join(frame_bindings)}
    >;"""

    per_scene_layout = material.gen_scene_layout()

    if per_scene_layout:
        layouts += \
f"""

    {per_scene_layout}"""

    return f"""template <>
struct PipelineProps<{specialization_type}> {{
    {sep.join(prop_members)};

    static constexpr const char *vertexShaderName =
        "{shader_name}.vert.spv";
    static constexpr const char *fragmentShaderName =
        "{shader_name}.frag.spv";

    {vertex.gen_attrs(specialization_type)}

    {sep.join(vert_input_locs)};

    {layouts}
}};
"""

def write_template_instantiations(type_str, v4r, v4r_display,
        render, loader, has_material):
    entry_instantiate = \
f"""template BatchRenderer::BatchRenderer(const RenderConfig &,
    const RenderFeatures<{type_str}> &);

template std::shared_ptr<Mesh> AssetLoader::loadMesh(
        std::vector<{type_str}::Vertex>,
        std::vector<uint32_t>);
"""

    if has_material:
        entry_instantiate += f"""
template std::shared_ptr<Material> AssetLoader::makeMaterial(
    {type_str}::MaterialParams);
"""
    
    print(entry_instantiate, file=v4r)

    print(
f"""template BatchPresentRenderer::BatchPresentRenderer(const RenderConfig &,
    const RenderFeatures<{type_str}> &, bool);
""", file=v4r_display)

    print(
f"""template VulkanState::VulkanState(const RenderConfig &,
    const RenderFeatures<{type_str}> &,
    const DeviceUUID &);

template VulkanState::VulkanState(const RenderConfig &,
    const RenderFeatures<{type_str}> &,
    CoreVulkanHandles &&);
""", file=render)

    loader_instantiate = \
f"""template LoaderImpl LoaderImpl::create<{type_str}::Vertex,
                                       {type_str}::MaterialParams>();

template std::shared_ptr<Mesh> LoaderState::makeMesh(
    std::vector<{type_str}::Vertex>,
    std::vector<uint32_t>);
"""

    if has_material:
        loader_instantiate += f"""
template std::shared_ptr<Material> LoaderState::makeMaterial(
    {type_str}::MaterialParams);
"""

    print(loader_instantiate, file=loader)

def generate_pipelines(cfg_file, interface_path, implementation_path, cmake):
    cfg = json.load(cfg_file)
    property_map = cfg['properties']
    param_definitions = add_default_definitions(cfg['param_types'])
    pipelines = cfg['pipelines']

    end = ';' if cmake else None

    config_file = open(os.path.join(interface_path,
        'config.inl'), 'w')

    specializations_file = open(os.path.join(interface_path,
        'specializations.inl'), 'w')

    render_defn_file = open(os.path.join(implementation_path,
        'render_definitions.inl'), 'w')
    
    loader_defn_file = open(os.path.join(implementation_path,
        'loader_definitions.inl'), 'w')

    render_inst_file = open(os.path.join(implementation_path,
        'render_instantiations.inl'), 'w')

    loader_inst_file = open(os.path.join(implementation_path,
        'loader_instantiations.inl'), 'w')

    entry_file = open(os.path.join(implementation_path,
        'v4r_instantiations.inl'), 'w')

    display_entry_file = open(os.path.join(implementation_path,
        'v4r_display_instantiations.inl'), 'w')

    all_files = (config_file, specializations_file, render_defn_file,
                 loader_defn_file, render_inst_file, loader_inst_file,
                 entry_file, display_entry_file)

    print("#include <cstring>\n", file=loader_defn_file)

    for f in all_files:
        print("namespace v4r {\n", file=f)

    for pipeline_name, pipeline_config in pipelines.items():
        pipeline_params = pipeline_config['params']
        valid_configs = pipeline_config['valid_configurations']
        common_props = set(pipeline_config['shader_properties'])

        combinations = generate_combinations(valid_configs, pipeline_params,
                                             param_definitions)

        cpp_pipeline_definition = generate_pipeline_definition(pipeline_name,
                pipeline_params)

        print(cpp_pipeline_definition, file=config_file)

        for param_config in combinations:
            iface = InterfaceTracker()
            iface.bind(0) # set 0 binding 0 is ViewInfo
            iface.bind(0) # set 0 binding 1 is LightingInfo
            sampler_bound = False
            
            shader_name = pipeline_name

            flag_defines = []
            props = common_props.copy()

            material = Material()

            for param_name, param_values in param_config.items():
                param_type = pipeline_params[param_name]["type"]

                shader_name += f"_{''.join([str(e) for e in param_values])}"

                for param_value in param_values:
                    if param_type == 'bool':
                        if param_value == True:
                            param_define = f"{param_name}".upper()
                        else:
                            continue
                    else:
                        param_define = f"{param_name}_{param_value}".upper()

                    if param_type == "DataSource":
                        attr_type = "".join(
                                [e.title() for e in param_name.split('_')])
                        attr_type += param_value

                        if param_value == "Texture":
                            if not sampler_bound:
                                iface.bind(1) # set 1 bind 0: texture sampler
                                sampler_bound = True

                            texture_bind_num = iface.bind(1)
                            param_bind_define = \
                                f"{param_define}_BIND={texture_bind_num}"
                            flag_defines.append(param_bind_define)

                            material.add_param(param_name,
                                    "std::shared_ptr<Texture>",
                                    "std::shared_ptr<Texture>",
                                    attr_type, True)
                        elif param_value == "Uniform":
                            num_components = pipeline_params[
                                    param_name]["num_components"]

                            if num_components == 1:
                                mat_type = "float"
                                qual_type = mat_type
                            else:
                                mat_type = f"vec{num_components}"
                                qual_type = f"glm::{mat_type}"

                            material.add_param(param_name, mat_type,
                                    qual_type, attr_type, False)

                    flag_defines.append(param_define)
                    props = props.union(param_definitions[param_type][
                        'values'][param_value])

                    props = props.union(pipeline_params[param_name].get(
                                    "extra_params") or [])

            flag_defines += filter(None,
                    [property_map[prop] for prop in props])

            pipeline_type = specialization_type(pipeline_name, param_config, 
                    pipeline_params)

            # Build vertex format
            vertex = build_vertex(props, iface)
            vertex_defn = vertex.gen_struct()
            vertex_impl = vertex.gen_impl(pipeline_type)

            flag_defines += vertex.get_defines()

            # Transform flags
            transform_input_location = iface.vert_in()
            flag_defines.append(f"TXFM1_LOC={transform_input_location}")
            flag_defines.append(f"TXFM2_LOC={iface.vert_in()}")
            flag_defines.append(f"TXFM3_LOC={iface.vert_in()}")

            if "needNormalMatrix" in props:
                flag_defines.append(f"NORMAL_TXFM1_LOC={iface.vert_in()}")
                flag_defines.append(f"NORMAL_TXFM2_LOC={iface.vert_in()}")
                flag_defines.append(f"NORMAL_TXFM3_LOC={iface.vert_in()}")

            # Miscellaneous flags
            if "needColorOutput" in props:
                flag_defines.append(f"COLOR_OUT_LOC={iface.frag_out()}")

            if "needDepthOutput" in props:
                flag_defines.append(f"DEPTH_LOC={iface.vert_out()}")
                flag_defines.append(f"DEPTH_OUT_LOC={iface.frag_out()}")

            if "needMaterial" in props:
                material_input_location = iface.vert_in()
                flag_defines.append(
                        f"MATERIAL_IN_LOC={material_input_location}")
                flag_defines.append(f"MATERIAL_LOC={iface.vert_out()}")
                
                packed_material = material.pack()

                material_defn = material.gen_struct()
                material_impl = material.gen_impl(pipeline_type,
                                                  packed_material)
            else:
                material_defn = "using MaterialParams = NoMaterial;"
                material_impl = None
                material_input_location = None
            
            if "needMaterialParams" in props:
                param_bind_num = iface.bind(1)
                material_param_bind = f"PARAM_BIND={param_bind_num}"
                flag_defines.append(material_param_bind)

                flag_defines += packed_material.gen_access_flags()
                flag_defines.append(
                    f"NUM_PARAM_VECS={packed_material.num_param_vecs()}")
                
            if "needLighting" in props:
                flag_defines.append(f"CAMERA_POS_LOC={iface.vert_out()}")

            shader_name = shader_name.lower()

            defines_str = " ".join([f"-D{define}" for define in flag_defines])
            print(f"{shader_name} {defines_str}", end=end)

            pipeline_specialization = generate_pipeline_specialization(
                    pipeline_type, vertex_defn, material_defn)

            props_specialization = generate_props_specialization(
                    pipeline_type, props, property_map.keys(),
                    vertex, material, shader_name,
                    transform_input_location,
                    material_input_location)

            print(pipeline_specialization, file=specializations_file)
            print(vertex_impl, file=loader_defn_file)
            if material_impl != None:
                print(material_impl, file=loader_defn_file)
            print(props_specialization, file=render_defn_file)

            write_template_instantiations(pipeline_type,
                    entry_file, display_entry_file,
                    render_inst_file, loader_inst_file,
                    "needMaterial" in props)

    for f in all_files:
        print("}", file=f)

    print('\n#include "specializations.inl"', file=config_file)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"{sys.argv[0]}: CONFIG_PATH INTERFACE_PATH \
IMPLEMENTATION_PATH [--cmake]",
              file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], 'r') as cfg_file:
        cmake = len(sys.argv) > 4 and sys.argv[4] == '--cmake'
        generate_pipelines(cfg_file, sys.argv[2],
                           sys.argv[3], cmake)
